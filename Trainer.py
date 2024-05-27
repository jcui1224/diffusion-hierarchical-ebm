import torch
import numpy as np
import torch.nn as nn
from nvae.utils import gather
import nvae.utils as utils
import torch.distributed as dist


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)


def stop_condition(tensor):
    return torch.isnan(tensor) or tensor.item() > 1e15 or tensor.item() < -1e15


class _DiffusionSchedule():
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.sigmas, self.a_s = self._get_sigma_schedule(
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=args.max_T
        )
        self.a_s_cum = np.cumprod(self.a_s)
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.copy()
        self.a_s_prev[-1] = 1
        self.is_recovery = np.ones(args.max_T + 1, dtype=np.float32)
        self.is_recovery[-1] = 0

    @staticmethod
    def _get_sigma_schedule(beta_start, beta_end, num_diffusion_timesteps):
        """
        Get the noise level schedule
        :param beta_start: begin noise level
        :param beta_end: end noise level
        :param num_diffusion_timesteps: number of timesteps
        :return:
        -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
        -- a_s: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
        """
        betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
        betas = np.append(betas, 1.)
        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        sqrt_alphas = np.sqrt(1. - betas)
        idx = torch.tensor(
            np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]]),
            dtype=torch.int32)
        a_s = np.concatenate(
            [[np.prod(sqrt_alphas[: idx[0] + 1])],
             np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
        sigmas = np.sqrt(1 - a_s ** 2)

        return sigmas, a_s

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(x_shape[0], dtype=torch.long, device=t.device) * t
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.tensor(a, dtype=torch.float32, device=t.device)[t]
        assert list(out.shape) == [bs]
        return out.reshape([bs] + ((len(x_shape) - 1) * [1]))

    def _q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(size=x_start.shape, device=t.device)

        assert noise.shape == x_start.shape
        x_t = self._extract(self.a_s_cum, t, x_start.shape) * x_start + \
              self._extract(self.sigmas_cum, t, x_start.shape) * noise

        return x_t

    def _q_sample_inc(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(size=x_start.shape, device=t.device)
        assert noise.shape == x_start.shape
        x_t = self._extract(self.a_s, t, x_start.shape) * x_start + \
              self._extract(self.sigmas, t, x_start.shape) * noise

        return x_t

    def _q_sample_pairs(self, x_start, t, noise=None):
        """
        Generate a pair of disturbed images for training
        :param x_start: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        x_t = self._q_sample(x_start, t, noise)
        if noise is None:
            noise = torch.randn(size=x_start.shape, device=t.device)

        x_t_plus_one = self._extract(self.a_s, t + 1, x_start.shape) * x_t + \
                       self._extract(self.sigmas, t + 1, x_start.shape) * noise
        return x_t, x_t_plus_one


class _DiffusionJointEBM():
    def __init__(self, z_list, ebm_list, args, logging):
        super().__init__()

        self.logging = logging
        self.max_T = args.max_T
        self.local_rank = args.local_rank
        self.args = args
        self.z_list = z_list
        self.ebm_list = ebm_list
        self.opt_list, self.lr_schedule_list = self.build_optimizer()
        self.schedule = self.build_diffusion_schedule()

    def build_diffusion_schedule(self):
        return _DiffusionSchedule(self.args)

    def build_optimizer(self):
        opt_list = []
        lr_schedule_list = []
        for i, netEzi in enumerate(self.ebm_list):
            opt = torch.optim.Adam(netEzi.parameters(), lr=self.args.e_lr, weight_decay=self.args.wd, betas=(0.99, 0.999))
            schedule = torch.optim.lr_scheduler.ExponentialLR(opt, self.args.e_gamma)
            opt_list.append(opt)
            lr_schedule_list.append(schedule)

        return opt_list, lr_schedule_list

    def get_posterior_eps(self, VAE, data):
        posterior_z = VAE.get_posterior_jcui7(data)
        s, posterior_eps = VAE.new_sample_eps_by_z(posterior_z)
        return s, posterior_eps

    def get_z_by_eps(self, VAE, eps_list=None, temp=1.0, ):
        bs = eps_list[0].shape[0]
        s, z_list = VAE.new_sample_z_by_eps(bs, temp, eps_list)
        return s, z_list

    def get_img_by_s(self, VAE, s):
        return VAE.new_sample_x_by_s(s).detach()

    @staticmethod
    def _set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=False for all the networks to
           avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks
                                     require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_flag(self):
        for ebm in self.ebm_list:
            ebm.train()

    def eval_flag(self):
        for ebm in self.ebm_list:
            ebm.eval()

    def tune_LD_step_size(self, t, shape):
        bs = shape[0]

        e_l_step_size = self.args.e_l_step_size
        sigma = self.schedule._extract(self.schedule.sigmas, t + 1, shape)
        sigma_cum = self.schedule._extract(self.schedule.sigmas_cum, t, shape)
        c_t_square = sigma_cum / self.schedule.sigmas_cum[0]
        e_l_step_size = c_t_square * e_l_step_size * (sigma ** 2)
        return e_l_step_size

    # === Sampling ===
    def _p_sample_langevin(self, VAE, tilde_x, t, display=False):
        """
        Langevin sampling function
        """
        bs = tilde_x[0].size(0)
        ys = [x.clone().detach().requires_grad_(True) for x in tilde_x]
        shape = tilde_x[0].shape

        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(bs, dtype=torch.long, device=tilde_x[0].device) * t

        e_l_steps = self.args.e_l_steps
        e_l_step_size = self.tune_LD_step_size(t, shape)

        sigma = self.schedule._extract(self.schedule.sigmas, t + 1, shape)
        a_s = self.schedule._extract(self.schedule.a_s, t + 1, shape)
        is_recovery = self.schedule._extract(self.schedule.is_recovery, t + 1, shape)

        for istep in range(e_l_steps):
            _, z_list = self.get_z_by_eps(VAE, ys)
            langevin_log = f" ======== LANGEVIN STEP {istep:2}/{e_l_steps} ===========\n"
            en_loss = 0.
            kernel_loss = 0.

            for i, (z, y, ebm) in enumerate(zip(z_list, ys, self.ebm_list)):
                en = ebm(z, t).squeeze(-1)
                kernel = torch.sum((y * a_s - tilde_x[i]) ** 2 / (2 * sigma ** 2) * is_recovery, dim=[1,2,3])
                gau_kernel = torch.sum((y ** 2) / 2, dim=[1,2,3])

                en_loss += en.sum(0)
                kernel_loss += (kernel.sum(0) + gau_kernel.sum(0))
                langevin_log += f"======== Z{i:2} Energy {en.mean().item():<10.3f} Kernel {kernel.mean().item():<2.3f}||\n "

            e_grad = torch.autograd.grad(en_loss, ys)
            g_grad = torch.autograd.grad(kernel_loss, ys)

            for i in range(len(ys)):
                noise = torch.randn_like(ys[i])
                ys[i].data = ys[i].data + 0.5 * e_grad[i] - 0.5 * e_l_step_size * g_grad[i] + torch.sqrt(e_l_step_size) * noise.data


            if (istep % 10 == 0 or istep == e_l_steps - 1) and display:
                self.logging.info(langevin_log)

        x = [y.detach() for i, y in enumerate(ys)]
        return x

    def p_sample_progressive(self, VAE, temp=1.0, noise=None):
        """
        Sample a sequence of images with the sequence of noise levels
        """
        if noise is None:
            noise = self.init_p_0()

        x_neg_t = [n.clone().detach() for n in noise]
        s, z_list = self.get_z_by_eps(VAE, x_neg_t, temp)
        syn = self.get_img_by_s(VAE, s)

        neg_list = [syn]
        for t in range(self.max_T - 1, -1, -1):
            x_neg_t = self._p_sample_langevin(VAE, x_neg_t, t)
            s, z_list = self.get_z_by_eps(VAE, x_neg_t, temp)
            syn = self.get_img_by_s(VAE, s)
            neg_list.append(syn)
        return neg_list[::-1]

    def init_p_0(self):
        noise_list = [torch.randn(zi.size()).cuda(self.local_rank) for zi in self.z_list]
        return noise_list

    def p_sample(self, VAE, temp=1.0, noise=None):
        """
        Sample a sequence of images with the sequence of noise levels
        """
        if noise is None:
            noise = self.init_p_0()
        x_neg_t = [n.clone().detach() for n in noise]
        for t in range(self.max_T - 1, -1, -1):
            x_neg_t = self._p_sample_langevin(VAE, x_neg_t, t)

        s, z_list = self.get_z_by_eps(VAE, x_neg_t, temp)
        syn = self.get_img_by_s(VAE, s)
        return syn

    # === Training ===
    def sample_latent_training_pairs(self, eps_list, t, fixed_noise=False):
        def q_sample_one_eps(eps, t_idx, noise):
            x_pos, x_neg = self.schedule._q_sample_pairs(eps, t_idx, noise)
            return x_pos, x_neg

        eps_pos_t, eps_neg_t_0 = [], []
        for i, eps in enumerate(eps_list):
            noise = torch.randn_like(eps, device=t.device) if fixed_noise else None
            x_pos, x_neg = q_sample_one_eps(eps, t, noise=noise)
            eps_pos_t.append(x_pos.detach())
            eps_neg_t_0.append(x_neg.detach())

        return eps_pos_t, eps_neg_t_0

    def get_loss(self, ebm, x_pos, x_neg, t):
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(x_pos.size(0), dtype=torch.long, device=x_pos.device) * t

        e_l_step_size = self.tune_LD_step_size(t, x_pos.shape)

        pos_f = ebm(x_pos, t)
        neg_f = ebm(x_neg, t)
        loss = - (pos_f - neg_f).squeeze(-1) / e_l_step_size.view(-1)

        return loss.mean(), pos_f.mean(), neg_f.mean()

    def train_fn(self, data, VAE, display=False):
        """
        posterior
        """
        self.train_flag()

        bs = data.shape[0]
        _, posterior_eps = self.get_posterior_eps(VAE, data)

        t = torch.randint(low=0, high=self.max_T, size=(bs,), device=data.device)

        x_pos_t, x_t_0 = self.sample_latent_training_pairs(posterior_eps, t)

        self._set_requires_grad(self.ebm_list, False)

        x_t_k = self._p_sample_langevin(VAE, x_t_0, t, display)

        _, x_pos_t = self.get_z_by_eps(VAE, x_pos_t)
        _, x_t_k = self.get_z_by_eps(VAE, x_t_k)

        self._set_requires_grad(self.ebm_list, True)

        for ebm in self.ebm_list:
            ebm.zero_grad()

        en_loss = 0.

        train_log = f'=========== Optimizing =========== \n'
        for i, (pos_t, neg_t) in enumerate(zip(x_pos_t, x_t_k)):
            ebm = self.ebm_list[i]
            loss_t, e_t, e_f = self.get_loss(ebm, pos_t, neg_t, t)
            loss = loss_t

            train_log += f"z{i:2}: {'err':<4}: {loss_t.item():<10.2f} {'e_t':<4}: {e_t.mean().item():<10.2f} {'e_f':<4}: {e_f.mean().item():<10.2f}\n"

            en_loss += loss

            loss_gather = [stop_condition(e) for e in gather(loss, nprocs=self.args.nprocs)]
            if True in loss_gather:
                return train_log, True

        en_loss.backward()
        dist.barrier()
        for ebm, opt in zip(self.ebm_list, self.opt_list):
            utils.average_gradients(ebm.parameters(), self.args.distributed)
            opt.step()

        self._set_requires_grad(self.ebm_list, False)
        self.eval_flag()
        return train_log, False

    def save_ckpt(self, dir):
        state_dict = {}
        for i in range(len(self.ebm_list)):
            state_dict[f'netE{i}'] = self.ebm_list[i].module.state_dict() if self.args.distributed else self.ebm_list[i].state_dict()
            state_dict[f'opt{i}'] = self.opt_list[i].state_dict()
        torch.save(state_dict, dir)

    def average_params(self):
        for netE in self.ebm_list:
            utils.average_params(netE.parameters(), self.args.distributed)
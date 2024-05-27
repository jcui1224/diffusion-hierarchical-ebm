from Trainer import _DiffusionJointEBM
import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import nvae.utils as utils
from nvae.utils import gather
from tqdm import tqdm
import random
import nvae.datasets
from utils import *
from matplotlib import image
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def visualize(LEBM, VAE, local_rank, global_step, args):
    imgs_dir = args.save + f'/images/{local_rank}/'
    os.makedirs(imgs_dir, exist_ok=True)

    syns = LEBM.p_sample_progressive(VAE, 1.0)

    def show(imgs, path):
        if isinstance(imgs[0], list):
            fig, axs = plt.subplots(nrows=len(imgs), ncols=len(imgs[0]), figsize=(25*len(imgs[0]), 25*len(imgs)))

            for i, img in enumerate(imgs):
                for j, im in enumerate(img):
                    im = image.imread(im)
                    axs[i, j].imshow(im)
                    axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            plt.savefig(path)
            plt.close()
        else:
            imgs = [imgs]
            fig, axs = plt.subplots(nrows=len(imgs), ncols=len(imgs[0]), figsize=(25 * len(imgs[0]), 25))

            for i, img in enumerate(imgs):
                for j, im in enumerate(img):
                    im = image.imread(im)
                    axs[j].imshow(im)
                    axs[j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            plt.savefig(path)
            plt.close()

    for i, syn in enumerate(syns):
        show_single_batch(syn, imgs_dir + f'current_syn_{i}.png')

    show([imgs_dir + f'current_syn_{i}.png' for i in range(len(syns))], path=imgs_dir + f'{global_step:>07d}.png')
    return


def get_fid(args, VAE, LEBM, z_list):
    from pytorch_fid_jcui7.fid_score import compute_fid
    fid_size = 30000
    batch_size = args.batch_size
    s = []
    n_batch = int((fid_size // batch_size) // args.nprocs)  # make sure it can divide

    def sample():
        syn = LEBM.p_sample(VAE.module)
        syn = syn.clamp(min=0., max=1.)
        return syn

    if args.local_rank == 0:  # print rank 0 progress
        count = tqdm(range(n_batch))
    else:
        count = range(n_batch)

    for _ in count:
        syn = sample()
        s.append(syn)

    s1 = torch.cat(s)

    dist.barrier()  # wait others

    s_gather = gather(s1, args.nprocs)

    fid_stat_dir = args.fid_stat_base_dir
    if args.local_rank == 0:
        s = torch.cat(s_gather, dim=0)
        fid1 = compute_fid(x_train=None, x_samples=s, path=fid_stat_dir)
        return fid1
    else:
        return math.inf


def train(VAE, z_list, LEBM, train_queue, logging, args):
    # for netE in LEBM.ebm_list:
    local_rank = args.local_rank

    global_step = 0
    fid_best = math.inf
    fid_best_ep = 0

    args.iter_metric = (len(train_queue)-10) // args.iter_metric
    print(args.iter_metric)

    for ep in range(args.epochs):

        for schedule in LEBM.lr_schedule_list:
            schedule.step(epoch=ep)

        train_queue.sampler.set_epoch(global_step)

        for b, x in enumerate(train_queue):
            global_step += 1
            x = x[0] if len(x) > 1 else x
            x = x.cuda(local_rank)

            if b % 10 == 0:  # just in case, maybe useless.
                LEBM.average_params()

            if b % args.log_iter == 0:
                logging.info("===" * 15 + f'Epoch {ep} Batch {b}/{len(train_queue)} fid best {fid_best} ep {fid_best_ep}' + "===" * 15)
                logging.info("===" * 15 + f'DIR {args.save}' + "===" * 15)

            train_log, broken_flag = LEBM.train_fn(x, VAE.module, display=(b % args.log_iter == 0))

            if broken_flag:
                print(f"Early Stopping: {train_log}")
                return

            if b % args.log_iter == 0:
                logging.info(train_log)

            if b > 0 and (b+1) % args.vis_iter == 0:
                visualize(LEBM, VAE.module, local_rank, global_step, args)

            dist.barrier()

            if ep >= args.fid_start_ep and b > 0 and args.compute_fid and b % args.iter_metric == 0:

                fid = get_fid(args, VAE, LEBM, z_list)
                if local_rank == 0:
                    logging.info(f"FID : {fid}")
                    if fid < fid_best:
                        fid_best = fid
                        fid_best_ep = ep
                        os.makedirs(args.save + f'/ckpt/', exist_ok=True)
                        LEBM.save_ckpt(args.save + f'/ckpt/EBM_{global_step}.pth')

                dist.barrier()  # wait rank

    return


def load_VAE(args, local_rank, logging):
    logging.info('loading the model at:')
    logging.info(args.checkpoint)

    from nvae.model import AutoEncoder
    import nvae.utils as utils
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    VAE_cfg = checkpoint['args']
    if not hasattr(VAE_cfg, 'num_mixture_dec'):
        VAE_cfg.num_mixture_dec = 10
    logging.info('loaded model at epoch %d', checkpoint['epoch'])

    arch_instance = utils.get_arch_cells(VAE_cfg.arch_instance)
    VAE = AutoEncoder(VAE_cfg, None, arch_instance)
    VAE = VAE.cuda(local_rank)

    VAE.load_state_dict(checkpoint['state_dict'], strict=False)
    VAE = torch.nn.parallel.DistributedDataParallel(VAE, device_ids=[local_rank])

    parameters = VAE.parameters()
    for p in parameters:
        p.requires_grad = False

    if args.bn:
        num_samples = args.batch_size
        iter = 500
        from torch.cuda.amp import autocast
        VAE.train()
        with autocast():
            for i in range(iter):
                VAE.module.sample(num_samples, 1.0)
    VAE.eval()
    return VAE


def build_EBM(z_list, args, local_rank, logging):
    from ebm_net import _netE_simple_res

    ebm_list = []
    for i, z in enumerate(z_list):
        shape = z.shape
        logging.info(f'z {i} shape: {shape}')
        netEzi = _netE_simple_res(shape, args).cuda(local_rank)
        netEzi = torch.nn.parallel.DistributedDataParallel(netEzi, device_ids=[local_rank])
        ebm_list.append(netEzi)

    return ebm_list


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main(local_rank, nprocs, args):
    # ==============================Prepare DDP ================================
    args.local_rank = local_rank
    init_seeds(local_rank)
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    print(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=nprocs, rank=local_rank)

    logging = utils.Logger(local_rank, args.save)
    logging.info(args)

    VAE = load_VAE(args, local_rank, logging)

    with torch.no_grad():  # get a bunch of samples to know how many groups of latent variables are there
        _, z_list = VAE.module.new_sample_z_by_eps(args.batch_size)

    ebm_list = build_EBM(z_list, args, local_rank, logging)

    LEBM = _DiffusionJointEBM(z_list, ebm_list, args, logging)
    train_queue, _, num_classes = nvae.datasets.get_loaders(args)

    train(VAE, z_list, LEBM, train_queue, logging, args)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--checkpoint', default='CELEBA256_NVAE_QUALI_DOWNLOADED')

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_iter", type=int, default=50)
    parser.add_argument("--vis_iter", type=int, default=50)
    parser.add_argument("--compute_fid", type=bool, default=True)
    parser.add_argument("--fid_start_ep", type=int, default=1)
    parser.add_argument("--iter_metric", type=int, default=4)

    parser.add_argument("--max_T", type=int, default=3)

    parser.add_argument("--save", type=str, default="", help="directory to store scores in")

    # ================= DDP ===================
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    # parser.add_argument('--batch_size', '--batch-size', default=4, type=int)# cifar
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--master_port', type=str, default='6029', help='port for master')
    parser.add_argument('--nprocs', type=int, default=2, help='number of gpus')
    args = parser.parse_args()

    from config import a100_celeba256 as base_config
    data_base_dir = base_config['data_base_dir']
    args.data = data_base_dir + 'celeba256_org/celeba256_lmdb'

    args = overwrite_opt(args, base_config)
    args.distributed = True

    output_dir = './{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_dir += f'{args.checkpoint}_{args.max_T}/{t}/'

    args.checkpoint = args.ckpt_base_dir + f'/{args.checkpoint}/checkpoint.pt'
    args.save = output_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'code/', exist_ok=True)

    [save_file(output_dir, f) for f in ['ebm_net.py', 'Trainer.py', 'utils.py', os.path.basename(__file__)]]

    save_args(vars(args), args.save + '/')

    mp.spawn(main, nprocs=args.nprocs, args=(args.nprocs, args))


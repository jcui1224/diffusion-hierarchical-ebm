import numpy as np
from collections import OrderedDict
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, max_period=10000):
        """
        :param emb_dim: embedding dimension
        :param max_len: maximum diffusion step
        """
        super(PositionalEmbedding, self).__init__()
        self.dim = emb_dim
        self.max_period = max_period

    def forward(self, t):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        dim = self.dim
        max_period = self.max_period
        timesteps = t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, temb, args):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv2d_1 = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        )

        self.conv2d_2 = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        )

        self.conv2d_shortcut = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        )

        self.linear = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(temb, out_ch, bias=False))
        )

    def forward(self, inputs, temb):
        x = inputs
        h = inputs

        h = self.conv2d_1(h) + self.linear(temb).view(-1, self.out_ch, 1, 1)
        h = self.conv2d_2(h)

        x = self.conv2d_shortcut(x)

        return x + h


class _CIFAR10_netE(nn.Module):
    def __init__(self, z_shape, args):
        super().__init__()
        self.z_shape = z_shape
        b, c, h, w = z_shape[0], z_shape[1], z_shape[2], z_shape[3]

        num_layers = {8:2, 16:3, 32:4, 64:5, 128:6}[h]
        self.ch = args.ch
        hidden = self.ch * 2 * 2
        # self.num_res_blocks = args.num_res_blocks
        self.max_T = args.max_T

        self.t_ebm = PositionalEmbedding(emb_dim=128)
        self.t_ebm_model1 = nn.Sequential(
            spectral_norm(nn.Linear(128, c, bias=False)),
            nn.SiLU(),
            spectral_norm(nn.Linear(c, c, bias=False)),
        )
        self.t_ebm_model2 = nn.Sequential(
            spectral_norm(nn.Linear(128, 128, bias=False)),
            nn.SiLU(),
            spectral_norm(nn.Linear(128, 128, bias=False)),
        )

        self.conv_start = spectral_norm(nn.Conv2d(c, self.ch, 4, 2, 1, bias=False))
        self.t_linear = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(128, self.ch, bias=False)),
        )
        self.conv_all = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Conv2d(self.ch, self.ch, 3, 1, 1, bias=False))
        )
        self.conv_short = spectral_norm(nn.Conv2d(c, self.ch, 4, 2, 1, bias=False))

        res_blocks = []
        for i in range(num_layers-1):
            res_blocks.append(conv_block(in_ch=self.ch, out_ch=self.ch, temb=128, args=args))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.linear1_x = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden, hidden // 2, bias=False)),
        )
        self.linear1_t = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(128, hidden // 2, bias=False)),
        )
        self.linear1_all = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden // 2, hidden // 2, bias=False)),
        )
        self.linear1_short = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden, hidden // 2, bias=False)),
        )

        self.linear2_x = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden // 2, hidden // 4, bias=False)),
        )
        self.linear2_t = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(128, hidden // 4, bias=False)),
        )
        self.linear2_all = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden // 4, hidden // 4, bias=False)),
        )
        self.linear2_short = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden // 2, hidden // 4, bias=False)),
        )

        self.out = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden // 4, 1, bias=False)),
        )

    def forward(self, inputs, t_idx):
        assert inputs.shape[1:] == self.z_shape[1:]
        bs = inputs.shape[0]
        t_idx = self.t_ebm(t_idx)
        t_input = self.t_ebm_model1(t_idx)
        t_h = self.t_ebm_model2(t_idx)

        inputs = inputs + t_input.view(inputs.shape[0], inputs.shape[1], 1, 1)

        f = self.conv_start(inputs) + self.t_linear(t_h).view(inputs.shape[0], self.ch, 1, 1)
        f = self.conv_all(f)
        f = f + self.conv_short(inputs)

        for block in self.res_blocks:
            f = block(f, t_h)

        f = f.view(bs, -1)

        x = f
        x = self.linear1_x(x) + self.linear1_t(t_h)
        f = self.linear1_all(x) + self.linear1_short(f)

        x = f
        x = self.linear2_x(x) + self.linear2_t(t_h)
        f = self.linear2_all(x) + self.linear2_short(f)

        en = self.out(f)
        return en


class _CelebA256_netE(nn.Module):
    def __init__(self, z_shape, args):
        super().__init__()
        self.z_shape = z_shape
        b, c, h, w = z_shape[0], z_shape[1], z_shape[2], z_shape[3]
        num_layers = {8:1, 16:2, 32:3, 64:4, 128:5}[h]
        self.ch = args.ch
        self.max_T = args.max_T

        self.t_ebm = PositionalEmbedding(emb_dim=128)
        self.t_ebm_model1 = nn.Sequential(
            spectral_norm(nn.Linear(128, 128, bias=False), mode=args.add_sn),
            nn.SiLU(),
            spectral_norm(nn.Linear(128, 128, bias=False), mode=args.add_sn),
        )

        convlayers = OrderedDict()
        current_dims = c
        for i in range(num_layers-1):
            convlayers['conv{}'.format(i + 1)] = spectral_norm(nn.Conv2d(current_dims, self.ch, 4, 2, 1, bias=False), mode=args.add_sn)
            convlayers['lrelu{}'.format(i + 1)] = nn.SiLU()
            current_dims = self.ch
        self.conv2d = nn.Sequential(convlayers)

        self.downsample = spectral_norm(nn.Conv2d(current_dims, self.ch, 4, 2, 1, bias=False), mode=args.add_sn)

        self.linear1_x = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4*4, self.ch*4*2, bias=False), mode=args.add_sn),
        )
        self.linear1_t = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(128, self.ch*4*2, bias=False), mode=args.add_sn),
        )
        self.linear1_all = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4*2, self.ch*4*2, bias=False), mode=args.add_sn),
        )
        self.linear1_short = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4*4, self.ch*4*2, bias=False), mode=args.add_sn),
        )

        self.linear2_x = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4*2, self.ch*4, bias=False), mode=args.add_sn),
        )
        self.linear2_t = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(128, self.ch*4, bias=False), mode=args.add_sn),
        )
        self.linear2_all = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4, self.ch*4, bias=False), mode=args.add_sn),
        )
        self.linear2_short = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4*2, self.ch*4, bias=False), mode=args.add_sn),
        )

        self.out = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(self.ch*4, 1, bias=False), mode=args.add_sn),
        )

    def forward(self, inputs, t_idx):
        assert inputs.shape[1:] == self.z_shape[1:]
        bs = inputs.shape[0]
        t_ = self.t_ebm_model1(self.t_ebm(t_idx))
        # inputs = inputs + t_.view(inputs.shape[0], inputs.shape[1], 1, 1)

        f = self.conv2d(inputs)
        f = self.downsample(f)
        f = f.view(bs, -1)

        x = f
        x = self.linear1_x(x) + self.linear1_t(t_)
        f = self.linear1_all(x) + self.linear1_short(f)

        x = f
        x = self.linear2_x(x) + self.linear2_t(t_)
        f = self.linear2_all(x) + self.linear2_short(f)

        en = self.out(f)
        return en
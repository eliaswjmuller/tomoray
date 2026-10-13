""" Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from vq_gan_3d.utils import shift_dim


class Codebook(pl.LightningModule):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # loaded codes are never re-initialised (resume, fine-tune, inside the DDPM)
        self._need_init = False

    @torch.no_grad()
    def nearest(self, flat_inputs, chunk=65536):
        """fp32 argmin in token chunks: bf16 flips near-tied assignments, and the full
        distance matrix (tokens x n_codes) was the training memory peak."""
        e = self.embeddings.float()
        e2 = (e ** 2).sum(dim=1)
        return torch.cat([((c ** 2).sum(dim=1, keepdim=True) - 2 * c @ e.t() + e2).argmin(dim=1)
                          for c in flat_inputs.split(chunk)])

    def snap(self, z):
        # z: [b, c, t, h, w] -> nearest code vectors, no codebook update
        with torch.autocast(z.device.type, enabled=False):
            idx = self.nearest(shift_dim(z.float(), 1, -1).flatten(end_dim=-2))
            return shift_dim(F.embedding(idx.view(z.shape[0], *z.shape[2:]), self.embeddings), -1, 1)

    def forward(self, z):
        with torch.autocast(z.device.type, enabled=False):
            return self._forward(z.float())

    def _forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)  # [bthw, c]
        flat_idx = self.nearest(flat_inputs)
        encoding_indices = flat_idx.view(z.shape[0], *z.shape[2:])  # [b, t, h, w]

        embeddings = F.embedding(
            encoding_indices, self.embeddings)  # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1)  # [b, c, t, h, w]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        counts = torch.bincount(flat_idx, minlength=self.n_codes).float()  # [ncode]

        # EMA codebook update
        if self.training:
            n_total = counts.clone()
            encode_sum = torch.zeros_like(self.embeddings).index_add_(0, flat_idx, flat_inputs)  # [ncode, c]
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum, alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1)
                         >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = counts / counts.sum()
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

class FSQ(nn.Module):
    """Finite scalar quantization (Mentzer et al. 2023): each channel is bounded to
    [-1, 1] by tanh and rounded to `levels` evenly spaced values. The implicit codebook
    is prod(levels) and every code is reachable, so nothing can collapse or die."""

    def __init__(self, levels):
        super().__init__()
        assert all(l % 2 == 1 for l in levels), f"odd levels keep 0 on the grid: {levels}"
        lv = torch.tensor(levels, dtype=torch.float64)
        self.register_buffer('half_width', ((lv - 1) / 2).float(), persistent=False)
        self.register_buffer('basis', torch.cumprod(torch.cat([lv.new_ones(1), lv[:-1]]), 0).long(),
                             persistent=False)

    def _h(self, z):
        return self.half_width.view(1, -1, *([1] * (z.dim() - 2)))

    def bound(self, z):
        return torch.tanh(z.float())

    def snap(self, zb):
        # zb: bounded latent [b, c, t, h, w] -> nearest grid value per channel
        h = self._h(zb)
        return torch.round(zb.float().clamp(-1, 1) * h) / h

    def forward(self, z):
        with torch.autocast(z.device.type, enabled=False):
            zb = self.bound(z)
            q = self.snap(zb)
            digits = (q * self._h(q) + self._h(q)).round().long()            # [b, c, t, h, w], 0..L-1
            idx = (digits * self.basis.view(1, -1, *([1] * (z.dim() - 2)))).sum(1)
            _, counts = idx.flatten().unique(return_counts=True)
            p = counts.float() / counts.sum()
            perplexity = torch.exp(-(p * torch.log(p)).sum())
        return dict(embeddings=zb + (q - zb).detach(), encodings=idx,
                    commitment_loss=zb.new_zeros(()), perplexity=perplexity)

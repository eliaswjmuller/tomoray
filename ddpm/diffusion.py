import math
import copy
import itertools
import os.path
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many


from rotary_embedding_torch import RotaryEmbedding
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from vq_gan_3d.model.vqgan import VQGAN
from evaluation.metrics import Metrics
from evaluation.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


# helpers functions

def normalize_for_metrics(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def volume_to_gif(tensor, path, duration=100, loop=0):
    """
    :param tensor: [C, D, H, W]
    """
    tensor = (tensor.clone().detach() + 1) + 0.5
    tensor = tensor.clamp(0, 1)

    images_tensors = tensor.unbind(dim=1) 
    
    to_pil = T.ToPILImage()
    images_pil = [to_pil(img) for img in images_tensors]

    if len(images_pil) > 0:
        images_pil[0].save(
            path, 
            save_all=True, 
            append_images=images_pil[1:],
            duration=duration, 
            loop=loop,
            optimize=False
        )


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5

# relative positional bias


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules

class Block(nn.Module):
    """
     block = 3D Conv + Group norm + (scale_shift) + SiLU
     scale_shift for time embedding

    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    """
     block = 3D Conv + Group norm + (scale_shift) + SiLU
     ResnetBlock = 2* block + residual connection
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,    # 64
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,  
        cond_channels=128,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8,
        cond_drop_prob=0.2
    ):
        super().__init__()

        # classifer free guidance

        self.cond_drop_prob = cond_drop_prob

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        concat_channels = channels + cond_channels

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(concat_channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding)) # replace channels by concat_channels

        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        # [64, 64, 128, 256]  ,  [64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]  # [64, 64, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(64, 64), (64, 128), (128, 256), (256, 512)]

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers 

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)  # replace cond_dim by time_dim

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):  # [(64, 64), (64, 128), (128, 256), (256, 512)]
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )


    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        
        return null_logits + (logits - null_logits) * cond_scale

    

    def forward(
        self,
        x,
        time,
        cond=None,
        cond_drop_prob=0.,
        prob_focus_present=0.,
        focus_present_mask=None


    ):
        
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))
        
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)


        if cond is not None and cond_drop_prob > 0:
            mask = prob_mask_like((batch,), cond_drop_prob, device=device)
            mask = rearrange(mask, 'b -> b 1 1 1 1')
            cond = cond * (1 - mask.float())
        
        # early fusion (concat with F_avg)
        if cond is not None:
            assert cond.shape[2:] == x.shape[2:], "Feature volume must match latent dimensions"
            x = torch.cat([x, cond], dim=1)

        
        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

# model


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels=8,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
        latent_mean=-0.1517,
        latent_std=2.3610,
    ):
        super().__init__()

        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        # Latent standardization. MUST match vqgan_ckpt: measured over 48 volumes of
        # the encoder output (quantize=False). The old codebook min/max normalization
        # gave std 0.072 (13.9x too small) because only ~220 of 4096 codes are live --
        # the dead ones sit far out in the tails and set the min/max.
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt, weights_only=False)
            self.vqgan.eval()

        else:
            self.vqgan = None
        
        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
    
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale))

        if clip_denoised:
            # NOTE: latents are now standardized (std 1), not squashed into [-1,1], so
            # this static s=1 clamp would cut ~32% of values. Only reached via
            # p_sample_loop/sample(), which nothing calls -- sample_dpm is the live path.
            # Enable use_dynamic_thres before using sample() again.
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=1):

        device = next(self.denoise_fn.parameters()).device


        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale)

        if isinstance(self.vqgan, VQGAN):
            _sample = _sample * self.latent_std + self.latent_mean
            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def sample_dpm(self, cond=None, cond_scale=1., batch_size=1, steps=20):

        device = self.betas.device

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        def unet_conditioned_wrapper(x, t, **kwargs):
            return self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        
        model_fn = model_wrapper(
            unet_conditioned_wrapper,
            noise_schedule,
            model_type="noise"
        )

        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')


        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames

        x_T = torch.randn((batch_size, channels, num_frames, image_size, image_size), device=device)

        x_sample = dpm_solver.sample(
            x_T,
            steps=steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

        if isinstance(self.vqgan, VQGAN):
            x_sample = x_sample * self.latent_std + self.latent_mean
            x_sample = self.vqgan.decode(x_sample, quantize=True)
        else:
            x_sample = unnormalize_img(x_sample)

        return x_sample
    
    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img


    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        """
        :param x: raw CT [B, 1, D, H, W]
        """
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                x = self.vqgan.encode(
                    x, quantize=False, include_embeddings=True)
                # standardize to ~N(0,1), which is what the noise schedule assumes
                x = (x - self.latent_mean) / self.latent_std
        else:
            print("Hi")
            x = normalize_img(x)

        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels,
                    f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, *args, **kwargs)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        fusion_model,
        cfg,
        accelerator=None,
        dataset=None,
        val_dataset=None,
        test_dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
        use_tensorboard=True,
        debug_overfit=True
    ):
        super().__init__()

        self.accelerator = accelerator
        self.is_main = self.accelerator.is_main_process if self.accelerator else True
        if self.accelerator:
            self.device = self.accelerator.device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = diffusion_model
        self.fusion_model = fusion_model

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model).to(self.device)

        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.cfg = cfg
        self.debug_overfit = debug_overfit

        self.writer = None
        if use_tensorboard and self.is_main:
            log_dir = os.path.join(results_folder, 'logs')
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"Logs saved in {log_dir}")

        assert dataset is not None, "Provide a dataset"
        self.ds = dataset

        if self.is_main:
            print(f'found {len(self.ds)} CTs.')
        assert len(self.ds) > 0, 'need to have at least 1 CT to start training'

        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,

        )
    
        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_dl = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        self.opt = Adam(
            list(diffusion_model.parameters()) + list(fusion_model.parameters()), 
            lr=train_lr
            )

        if self.accelerator:
            # val/test loaders stay unprepared: validation runs on the main process
            # only, and a sharded loader would score just 1/num_processes of the set.
            self.model, self.fusion_model, self.opt, dl = self.accelerator.prepare(
                self.model, self.fusion_model, self.opt, dl
            )
        else:
            self.model = self.model.to(self.device)
            self.fusion_model = self.fusion_model.to(self.device)

        self.val_dl = val_dl
        self.test_dl = test_dl
        self.metrics = Metrics(results_folder, val_dl, device=self.device)

        if self.debug_overfit:
            print("Debug mode:")

            self.static_batch = next(iter(dl))

            def infinite_static_loader():
                while True:
                    yield self.static_batch
            
            self.dl = infinite_static_loader()
        else:
            self.dl = cycle(dl)


        self.len_dataloader = len(dl)
        
        self.step = 0

        self.amp = amp and self.device.type == "cuda"
        # bf16, matching Accelerator(mixed_precision="bf16"); torch's autocast
        # default on cuda is fp16, which overflows the vqgan/loss path.
        self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler(self.device.type, enabled=self.amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def trainable_parameters(self):
        """Everything the optimizer owns -- diffusion U-Net AND the Fusion U-Net.
        Clipping only self.model left the Fusion grads unbounded."""
        return list(itertools.chain(self.model.parameters(), self.fusion_model.parameters()))

    def reset_parameters(self):
        unwrapped_model = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        self.ema_model.load_state_dict(unwrapped_model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        self.ema.update_model_average(self.ema_model, unwrapped_model)

    def save(self, milestone):

        if not self.is_main:
            return
        
        unwrapped_model = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        unwrapped_fusion = self.accelerator.unwrap_model(self.fusion_model) if self.accelerator else self.fusion_model

        data = {
            'step': self.step,
            'model': unwrapped_model.state_dict(),
            'fusion': unwrapped_fusion.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),   # without this a resume restarts Adam cold
            'scaler': self.scaler.state_dict()
        }
        path_cp = os.path.join(self.results_folder, 'checkpoints')
        os.makedirs(path_cp, exist_ok=True)
        path = os.path.join(path_cp, f'sample-{milestone}.pt')
        torch.save(data, path)

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_paths = os.listdir(os.path.join(self.results_folder, 'checkpoints'))
            all_milestones = [int((p.split('.')[0]).split("-")[-1]) for p in all_paths if p.endswith('.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location is None:
            map_location = self.device

        if isinstance(milestone, str) or isinstance(milestone, Path):
            ckpt_path = milestone
        else:
            ckpt_path = os.path.join(self.results_folder, 'checkpoints', f"sample-{milestone}.pt")

        print('found checkpoint', ckpt_path)
        data = torch.load(ckpt_path, map_location=map_location)

        self.step = data['step']

        unwrap = self.accelerator.unwrap_model if self.accelerator else (lambda m: m)
        unwrapped_model = unwrap(self.model)
        unwrapped_fusion = unwrap(self.fusion_model)
        unwrapped_ema = unwrap(self.ema_model)

        # strict=True: a silent partial load looks exactly like a fresh divergence
        unwrapped_model.load_state_dict(data['model'])
        unwrapped_fusion.load_state_dict(data['fusion'])
        unwrapped_ema.load_state_dict(data['ema'])

        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])
        else:
            print("WARNING: checkpoint has no optimizer state, Adam moments restart from zero")

        self.scaler.load_state_dict(data['scaler'])

        print("checkpoint is successful loaded")

    def save_image(self, image_tensor, path, cols=3):
        B, C, H, W = image_tensor.shape
        plt.figure(figsize=(50, 50))
        for i in range(B):
            plt.subplot(B // cols + 1, cols, i + 1)
            img = image_tensor[i].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img, cmap='gray' if C == 1 else None)
            plt.axis('off')
        plt.savefig(path)
        plt.close()

    def save_comparison(self, real, fake, path):
        # Fixed [0,1] window on BOTH panels. Without vmin/vmax matplotlib autoscales
        # each panel to its own extrema, so a washed-out sample renders at full
        # contrast and looks far better than it is.
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Real CT")
        plt.imshow(real.numpy(), cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Generated [{fake.min():.2f},{fake.max():.2f}]")
        plt.imshow(fake.numpy(), cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        
        plt.savefig(path)
        plt.close()

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        assert self.train_num_steps >= self.step_start_ema, 'num_steps must be greater that start_ema'
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)

                img = batch['image'].to(self.device)
                xrays = batch['projections'].to(self.device)
                angles = batch['angles'].to(self.device)

                with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                    
                    cond = self.fusion_model(xrays, angles)

                    if self.step % self.save_and_sample_every == 0 and self.is_main:

                        debug_dir = self.results_folder / 'debug_fusion_maps'
                        debug_dir.mkdir(exist_ok=True)

                        with torch.no_grad():
                            fusion_features = cond.detach().cpu()
                            
                            d_mid = fusion_features.shape[2] // 2
                            h_mid = fusion_features.shape[3] // 2
                            w_mid = fusion_features.shape[4] // 2

                            slice_axial = fusion_features[:, :, d_mid, :, :].mean(dim=1, keepdim=True)
                            slice_coronal = fusion_features[:, :, :, h_mid, :].mean(dim=1, keepdim=True)
                            slice_sagittal = fusion_features[:, :, :, :, w_mid].mean(dim=1, keepdim=True)

                            def save_norm(tensor, name):
                                        vmin, vmax = tensor.min(), tensor.max()
                                        tensor = (tensor - vmin) / (vmax - vmin + 1e-8)
                                        self.save_image(tensor, str(debug_dir / f'{name}_{self.step}.png'))

                            save_norm(slice_axial, 'fusion_axial')
                            save_norm(slice_coronal, 'fusion_coronal')
                            save_norm(slice_sagittal, 'fusion_sagittal')



                    if self.debug_overfit:
                        current_cond_drop = 0.0
                    else:
                        current_cond_drop = 0.1

                    loss = self.model(
                        img,
                        cond=cond,
                        # classifier free guidance
                        cond_drop_prob=current_cond_drop
                    )

                if self.accelerator:
                    self.accelerator.backward(loss / self.gradient_accumulate_every)
                else:
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            if self.step % 10 == 0 and self.is_main:
                print(f'{self.step}: {loss.item()}')

                if self.writer:
                    self.writer.add_scalar('loss/train', loss.item(), self.step)

            if self.is_main:
                log_fn({'loss': loss.item()})

            if self.accelerator:
                if exists(self.max_grad_norm):
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.trainable_parameters(), self.max_grad_norm)
                else:
                    grad_norm = None
                # Clipping does NOT stop a NaN: a non-finite total norm makes the clip
                # coefficient non-finite and poisons every parameter. One bad step then
                # corrupts Adam's moments permanently, which is how the run to step 12k
                # went to NaN and never recovered. Skip the step instead.
                # Decide on grad_norm, not loss: grads are already all-reduced, so it is
                # identical on every rank (and non-finite if ANY rank went bad), whereas
                # per-rank loss would let ranks disagree and silently desync DDP.
                if grad_norm is None:
                    flag = torch.tensor(
                        [0.0 if torch.isfinite(loss) else 1.0], device=self.device)
                    grad_norm = self.accelerator.reduce(flag, reduction="sum")
                if not torch.isfinite(grad_norm).all():
                    if self.is_main:
                        print(f'{self.step}: non-finite loss/grad, step skipped '
                              f'(loss={loss.item()}, grad_norm={grad_norm})')
                    self.opt.zero_grad()
                else:
                    self.opt.step()
                    self.opt.zero_grad()
            else:
                if exists(self.max_grad_norm):
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        self.trainable_parameters(), self.max_grad_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0 and self.is_main:
                self.ema_model.eval()
                self.fusion_model.eval()

                with torch.no_grad():

                    print("Calculating Validation Loss..")
                    total_val_loss = 0.0
                    for val_batch in self.val_dl:
                        val_img = val_batch["image"].to(self.device)
                        val_xrays = val_batch["projections"].to(self.device)
                        val_angles = val_batch["angles"].to(self.device)

                        with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                            val_cond = self.fusion_model(val_xrays, val_angles)
                            val_loss = self.ema_model(
                                val_img,
                                cond=val_cond,
                                cond_drop_prob=0.0
                            )
                        
                        total_val_loss+= val_loss.item()

                    avg_val_loss = total_val_loss / len(self.val_dl)
                    print(f"Validation Loss at step {self.step}: {avg_val_loss:.4f}")

                    if self.writer:
                        self.writer.add_scalar('loss/val', avg_val_loss, self.step)

                    milestone = self.step // self.save_and_sample_every
                    print(f"Sampling for milestone {milestone}...")

                    train_sample_dir = self.results_folder / 'train_samples'
                    val_sample_dir = self.results_folder / 'val_samples'
                    train_sample_dir.mkdir(exist_ok=True)
                    val_sample_dir.mkdir(exist_ok=True)

                    # Inference on training set
                    sample_xrays = xrays[:1]
                    sample_angles = angles[:1]
                    real_img = img[:1]
                    
                    sample_cond = self.fusion_model(sample_xrays, sample_angles)

                    all_samples = self.ema_model.sample_dpm(
                        cond=sample_cond,
                        cond_scale=2.0,
                        batch_size = 1,
                        steps=20
                    )

                    mid = all_samples.shape[2] // 2
                    gen_slice = all_samples[0, 0, mid, :, :].cpu()
                    real_slice = real_img[0, 0, mid, :, :].cpu()

                    gen_slice = (gen_slice + 1) * 0.5
                    real_slice = (real_slice + 1) * 0.5 
                    
                    self.save_comparison(
                        real_slice,
                        gen_slice,
                        str(train_sample_dir / f'sample-{milestone}.png')
                    )

                    if self.writer:
                        self.writer.add_image("Train/GT", real_slice.unsqueeze(0), self.step)
                        self.writer.add_image("Train/Generated", gen_slice.unsqueeze(0), self.step)


                    gen_vol_np = all_samples[0, 0].cpu().numpy()
                    real_vol_np = real_img[0, 0].cpu().numpy()

                    self.metrics.save_gif(real_vol=real_vol_np, fake_vol=gen_vol_np, milestone=milestone, phase='train')

                    # Inference on validation set
                    val_sample_batch = next(iter(self.val_dl))
                    sample_xrays_val = val_sample_batch["projections"][:1].to(self.device)
                    sample_angles_val = val_sample_batch["angles"][:1].to(self.device)
                    real_img_val = val_sample_batch["image"][:1].to(self.device)

                    sample_cond_val = self.fusion_model(sample_xrays_val, sample_angles_val)
                    gen_val = self.ema_model.sample_dpm(
                        cond=sample_cond_val, cond_scale=2.0, batch_size=1, steps=20
                    )

                    mid_v = gen_val.shape[2] // 2
                    gen_slice_v = (gen_val[0, 0, mid_v, :, :].cpu() + 1) * 0.5
                    real_slice_v = (real_img_val[0, 0, mid_v, :, :].cpu() + 1) * 0.5

                    self.save_comparison(
                        real_slice_v, gen_slice_v, 
                        str(val_sample_dir / f'sample-{milestone}.png')
                    )

                    if self.writer:
                        self.writer.add_image("Val/GT", real_slice_v.unsqueeze(0), self.step)
                        self.writer.add_image("Val/Generated", gen_slice_v.unsqueeze(0), self.step)

                    gen_val_np = gen_val[0, 0].cpu().numpy()
                    real_val_np = real_img_val[0, 0].cpu().numpy()
                    
                    self.metrics.save_gif(real_vol=real_val_np, fake_vol=gen_val_np, milestone=milestone, phase="val")
                    
                    # DISABLED: this ran on rank 0 only, inside `if self.is_main`, doing
                    # full sampling + LPIPS over all 1717 val volumes. It took >1h, so
                    # ranks 1/2 blocked at the next collective and NCCL's watchdog
                    # (timeout 3600s) killed the job at step 5000 -- its first ever call
                    # (milestone % 5). Score checkpoints offline instead; nothing here
                    # needs metrics to keep training.
                    # if milestone % 5 == 0:
                    #     self.metrics.update_metrics(self.ema_model, self.fusion_model, self.step)
                    
                    
                    if self.writer:
                        self.writer.add_image("GT", real_slice.unsqueeze(0), self.step)
                        self.writer.add_image("Generated", gen_slice.unsqueeze(0), self.step)
                
                self.save(milestone)
                self.fusion_model.train()
            
            self.step += 1
            
        if self.is_main:
            print('Training completed')

    @torch.inference_mode()
    def test(self, milestone):

        self.load(milestone)
        self.ema_model.eval()
        self.fusion_model.eval()

        if self.is_main:
            print(f"Loading test :")

        total_test_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        test_sample_dir = self.results_folder / 'test_samples'
        if self.is_main:
            test_sample_dir.mkdir(exist_ok=True)

        for i, test_batch in enumerate(tqdm(self.test_dl, desc="Test")):
            test_img = test_batch["image"].to(self.device)
            test_xrays = test_batch["projections"].to(self.device)
            test_angles = test_batch["angles"].to(self.device)

            with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                test_cond = self.fusion_model(test_xrays, test_angles)
                
                test_loss = self.ema_model(
                    test_img,
                    cond=test_cond,
                    cond_drop_prob=0.0 
                )
            total_test_loss += test_loss.item()

            gen_test = self.ema_model.sample_dpm(
                cond=test_cond, 
                cond_scale=2.0, 
                batch_size=test_img.shape[0], 
                steps=20
            )

            input1 = normalize_for_metrics(gen_test)
            input2 = normalize_for_metrics(test_img)

            psnr_val = self.metrics.psnr_3d(input1, input2)
            ssim_val, _ = self.metrics.ssim_3d(input1, input2)

            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()

            if self.is_main:
                tqdm.write(f"▶ CT {i+1} | PSNR: {psnr_val.item():.2f} dB | SSIM: {ssim_val.item():.4f}")

            if self.is_main:
                mid_z = input1.shape[2] // 2
                mid_y = input1.shape[3] // 2
                mid_x = input1.shape[4] // 2
                

                gen_slice_axial = input1[0, 0, mid_z, :, :].cpu()
                real_slice_axial = input2[0, 0, mid_z, :, :].cpu()
                self.save_comparison(
                    real_slice_axial, gen_slice_axial, 
                    str(test_sample_dir / f'test-sample-{i}-1_axial.png')
                )

                gen_slice_coronal = input1[0, 0, :, mid_y, :].cpu()
                real_slice_coronal = input2[0, 0, :, mid_y, :].cpu()
                self.save_comparison(
                    real_slice_coronal, gen_slice_coronal, 
                    str(test_sample_dir / f'test-sample-{i}-2_coronal.png')
                )

                gen_slice_sagittal = input1[0, 0, :, :, mid_x].cpu()
                real_slice_sagittal = input2[0, 0, :, :, mid_x].cpu()
                self.save_comparison(
                    real_slice_sagittal, gen_slice_sagittal, 
                    str(test_sample_dir / f'test-sample-{i}-3_sagittal.png')
                )
                
                gen_test_np = gen_test[0, 0].cpu().numpy()
                real_test_np = test_img[0, 0].cpu().numpy()
                
                self.metrics.save_gif(
                    real_vol=real_test_np, 
                    fake_vol=gen_test_np, 
                    milestone=f"test_{i}", 
                    phase="test"
                )

        num_batches = len(self.test_dl)
        avg_test_loss = total_test_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        if self.is_main:
            print(f"Test loss  : {avg_test_loss:.4f}")
            print(f"Mean PSNR : {avg_psnr:.2f} dB")
            print(f"Mean SSIM : {avg_ssim:.4f}")

        return avg_test_loss, avg_psnr, avg_ssim

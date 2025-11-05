# sample_ldm_suite.py
# Sample from: VAE, VQ-VAE, LDM(VAE path), LDM(VQ path)
# - Standalone: reproduces the exact decoders/UNet + schedule you used
# - Robust ckpt discovery: falls back to {out_dir}/vae/vae_last.pt etc.
# - Skip switches: --skip_vae --skip_vqvae --skip_ldm_vae --skip_ldm_vq
# - Sampler: --sampler ddpm|ddim (default ddpm), --ddim_steps controls DDIM
# ---------------------------------------------------------------------------------

import os, math, argparse, random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


# =============================
# Utilities
# =============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for speed we keep cudnn benchmark True; determinism not required for sampling grids
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_grid_img(tensor: torch.Tensor, path: str, nrow: int = 8, normalize=True, value_range=(-1, 1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, path)
    print(f"[Saved] {path}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_gn(ch: int, max_groups: int = 32):
    g = math.gcd(ch, max_groups)
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, ch)


# =============================
# VAE / VQ-VAE decoders (matching your training code)
# =============================

class ResBlockSimple(nn.Module):
    def __init__(self, ch, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            make_gn(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.Dropout(drop),
            make_gn(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class VAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base=128, drop=0.1):
        super().__init__()
        # Encoder (not needed for sampling, but kept for strict ckpt load)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base, 3, stride=2, padding=1),  # 32->16
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),  # 16->8
            ResBlockSimple(base * 2, drop),
        )
        self.to_mu = nn.Conv2d(base * 2, latent_ch, 1)
        self.to_logvar = nn.Conv2d(base * 2, latent_ch, 1)

        # Decoder
        self.dec_in = nn.Conv2d(latent_ch, base * 2, 1)
        self.dec = nn.Sequential(
            ResBlockSimple(base * 2, drop),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),  # 8->16
            ResBlockSimple(base, drop),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),      # 16->32
            ResBlockSimple(base, drop),
            make_gn(base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    # Only need decode() for sampling
    def decode(self, z):
        return self.dec(self.dec_in(z))


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed=512, embed_dim=4, beta=0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, C)
        emb = self.embedding.weight
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ emb.t()
            + emb.pow(2).sum(1, keepdim=True).t()
        )
        _, idx = torch.min(dist, dim=1)
        z_q = emb[idx].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commit_loss = F.mse_loss(z_q, z_e.detach())
        loss = codebook_loss + self.beta * commit_loss
        return z_q_st, loss, idx.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base=128, drop=0.1, n_embed=512, beta=0.25):
        super().__init__()
        # Encoder (kept to load ckpt strictly)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base, 3, stride=2, padding=1),  # 32->16
            ResBlockSimple(base, drop),
            nn.Conv2d(base, latent_ch, 3, stride=2, padding=1),  # 16->8
            ResBlockSimple(latent_ch, drop),
        )
        self.quant = VectorQuantizer(n_embed=n_embed, embed_dim=latent_ch, beta=beta)
        self.dec = nn.Sequential(
            ResBlockSimple(latent_ch, drop),
            nn.ConvTranspose2d(latent_ch, base, 4, stride=2, padding=1),  # 8->16
            ResBlockSimple(base, drop),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),       # 16->32
            ResBlockSimple(base, drop),
            make_gn(base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    # For VQ path we will call quant.embedding & dec directly for pure sampling
    def decode(self, z_q):
        return self.dec(z_q)


# =============================
# UNet (CompVis-style) & diffusion schedule (matching your training code)
# =============================

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def normalization(ch):
    g = math.gcd(ch, 32)
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, ch)


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, ch, use_conv=True):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(self, ch, emb_ch, dropout, out_ch=None, use_conv=False):
        super().__init__()
        out_ch = out_ch or ch
        self.in_layers = nn.Sequential(normalization(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch))
        self.out_layers = nn.Sequential(normalization(out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Identity() if out_ch == ch else (nn.Conv2d(ch, out_ch, 3, padding=1) if use_conv else nn.Conv2d(ch, out_ch, 1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip(x) + h


class UNetCompVis(nn.Module):
    def __init__(self, in_ch=4, model_ch=128, out_ch=4, channel_mult=(1, 2, 2, 2), num_res_blocks=2, dropout=0.1, conv_resample=True):
        super().__init__()
        self.model_ch = model_ch
        self.time_dim = model_ch * 4

        self.time_embed = nn.Sequential(nn.Linear(model_ch, self.time_dim), nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_ch, model_ch, 3, padding=1))])
        input_block_chans = [model_ch]
        ch = model_ch

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, self.time_dim, dropout, out_ch=mult * model_ch)]
                ch = mult * model_ch
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, use_conv=conv_resample)))
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(ResBlock(ch, self.time_dim, dropout), ResBlock(ch, self.time_dim, dropout))

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, self.time_dim, dropout, out_ch=model_ch * mult)]
                ch = model_ch * mult
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=conv_resample))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(normalization(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))

    def forward(self, x, t):
        emb = timestep_embedding(t, self.model_ch)
        emb = self.time_embed(emb)
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)


class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        return (x_t - sqrt_1mab * eps) / (sqrt_ab + 1e-8)

    @torch.no_grad()
    def sample_ddpm(self, model: UNetCompVis, shape: Tuple[int, int, int, int], device="cpu"):
        B, C, H, W = shape
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            betas_t = self.betas[t].view(-1, 1, 1, 1)
            eps = model(x, t)
            if step > 0:
                noise = torch.randn_like(x)
                mean = (1.0 / torch.sqrt(1.0 - betas_t)) * (
                    x - betas_t / torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1, 1) * eps
                )
                x = mean + torch.sqrt(betas_t) * noise
            else:
                x = self.predict_x0_from_eps(x, t, eps)
        return x

    @torch.no_grad()
    def sample_ddim(self, model: UNetCompVis, shape: Tuple[int, int, int, int], steps=50, device="cpu"):
        B, C, H, W = shape
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i in range(steps):
            t = ts[i].repeat(B)
            eps = model(x, t)
            x0 = self.predict_x0_from_eps(x, t, eps)
            if i == steps - 1:
                x = x0
                break
            t_next = ts[i + 1].repeat(B)
            ab_t_next = self.alpha_bars[t_next].view(-1, 1, 1, 1)
            x = torch.sqrt(ab_t_next) * x0 + torch.sqrt(1 - ab_t_next) * eps
        return x


# =============================
# Sampling helpers
# =============================

@torch.no_grad()
def sample_from_vae(vae: VAE, n: int, latent_ch: int, h: int, w: int, device, out_path: str, nrow: int = 8):
    vae.eval()
    z = torch.randn(n, latent_ch, h, w, device=device)
    imgs = vae.decode(z).clamp(-1, 1).cpu()
    save_grid_img(imgs, out_path, nrow=nrow)


@torch.no_grad()
def sample_from_vqvae(vq: VQVAE, n: int, latent_ch: int, h: int, w: int, device, out_path: str, nrow: int = 8):
    vq.eval()
    # Sample code indices uniformly, then embed and decode
    idx = torch.randint(low=0, high=vq.quant.n_embed, size=(n, h, w), device=device)
    emb = vq.quant.embedding(idx.view(-1))  # (n*h*w, C)
    z_q = emb.view(n, h, w, latent_ch).permute(0, 3, 1, 2).contiguous()
    imgs = vq.decode(z_q).clamp(-1, 1).cpu()
    save_grid_img(imgs, out_path, nrow=nrow)


@torch.no_grad()
def sample_from_ldm_with_vae(unet: UNetCompVis, vae: VAE, sched: DiffusionSchedule,
                             n: int, latent_ch: int, h: int, w: int, device, out_path: str, nrow: int,
                             sampler: str, ddim_steps: int):
    unet.eval(); vae.eval()
    shape = (n, latent_ch, h, w)
    if sampler == "ddim":
        latents = sched.sample_ddim(unet, shape, steps=ddim_steps, device=device)
    else:
        latents = sched.sample_ddpm(unet, shape, device=device)
    imgs = vae.decode(latents).clamp(-1, 1).cpu()
    save_grid_img(imgs, out_path, nrow=nrow)


@torch.no_grad()
def sample_from_ldm_with_vq(unet: UNetCompVis, vq: VQVAE, sched: DiffusionSchedule,
                            n: int, latent_ch: int, h: int, w: int, device, out_path: str, nrow: int,
                            sampler: str, ddim_steps: int):
    unet.eval(); vq.eval()
    shape = (n, latent_ch, h, w)
    if sampler == "ddim":
        z_e = sched.sample_ddim(unet, shape, steps=ddim_steps, device=device)
    else:
        z_e = sched.sample_ddpm(unet, shape, device=device)
    # Quantize AFTER diffusion (your training used this convention for VQ path)
    z_q, _, _ = vq.quant(z_e)
    imgs = vq.decode(z_q).clamp(-1, 1).cpu()
    save_grid_img(imgs, out_path, nrow=nrow)


def auto_ckpt(base: str, rel: str) -> Optional[str]:
    path = os.path.join(base, rel)
    return path if os.path.isfile(path) else None


# =============================
# Argparse / main
# =============================

def parse_args():
    p = argparse.ArgumentParser("Sample 32 images from VAE, VQ-VAE, and LDM (VAE/VQ paths)")

    # IO & device
    p.add_argument("--out_dir", type=str, default="outputs_ldm_toy_compvis", help="Root folder containing subdirs / checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--nrow", type=int, default=8)

    # Image / latent geometry
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--ae_base", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    # VQ-VAE specifics
    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--vq_beta", type=float, default=0.25)

    # UNet arch
    p.add_argument("--unet_model_ch", type=int, default=128)
    p.add_argument("--unet_channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--unet_num_res_blocks", type=int, default=2)
    p.add_argument("--no_conv_resample", action="store_true")

    # Diffusion schedule & sampler
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--ddim_steps", type=int, default=50)

    # Explicit ckpt overrides (optional)
    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--vqvae_ckpt", type=str, default=None)
    p.add_argument("--unet_vae_ckpt", type=str, default=None)
    p.add_argument("--unet_vq_ckpt", type=str, default=None)

    # Skips
    p.add_argument("--skip_vae", action="store_true")
    p.add_argument("--skip_vqvae", action="store_true")
    p.add_argument("--skip_ldm_vae", action="store_true")
    p.add_argument("--skip_ldm_vq", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Geometry
    H = W = args.img_size // 4  # matches your enc downsampling (32 -> 8)
    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-discover ckpts if not provided
    vae_ckpt = args.vae_ckpt or auto_ckpt(args.out_dir, "vae/vae_last.pt")
    vq_ckpt  = args.vqvae_ckpt or auto_ckpt(args.out_dir, "vqvae/vqvae_last.pt")
    u_vae_ck = args.unet_vae_ckpt or auto_ckpt(args.out_dir, "ldm_vae/unet_last.pt")
    u_vq_ck  = args.unet_vq_ckpt or auto_ckpt(args.out_dir, "ldm_vq/unet_last.pt")

    # Build schedule once
    sched = DiffusionSchedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    # ---- VAE ----
    if not args.skip_vae:
        if vae_ckpt and os.path.isfile(vae_ckpt):
            vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base, drop=args.dropout).to(device)
            vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
            print(f"[Load] VAE ckpt: {vae_ckpt} | params={count_params(vae):,}")
            out_path = os.path.join(args.out_dir, "samples_vae.png")
            sample_from_vae(vae, n=args.n, latent_ch=args.latent_channels, h=H, w=W, device=device, out_path=out_path, nrow=args.nrow)
        else:
            print("[Skip] VAE: checkpoint not found")
    else:
        print("[Skip] VAE: requested")

    # ---- VQ-VAE ----
    if not args.skip_vqvae:
        if vq_ckpt and os.path.isfile(vq_ckpt):
            vq = VQVAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base,
                       drop=args.dropout, n_embed=args.codebook_size, beta=args.vq_beta).to(device)
            vq.load_state_dict(torch.load(vq_ckpt, map_location=device))
            print(f"[Load] VQ-VAE ckpt: {vq_ckpt} | params={count_params(vq):,}")
            out_path = os.path.join(args.out_dir, "samples_vqvae.png")
            sample_from_vqvae(vq, n=args.n, latent_ch=args.latent_channels, h=H, w=W, device=device, out_path=out_path, nrow=args.nrow)
        else:
            print("[Skip] VQ-VAE: checkpoint not found")
    else:
        print("[Skip] VQ-VAE: requested")

    # ---- LDM (VAE path) ----
    if not args.skip_ldm_vae:
        if u_vae_ck and vae_ckpt and os.path.isfile(u_vae_ck) and os.path.isfile(vae_ckpt):
            # decoder VAE
            vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base, drop=args.dropout).to(device)
            vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
            # UNet
            unet_vae = UNetCompVis(
                in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
                channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
                dropout=args.dropout, conv_resample=not args.no_conv_resample
            ).to(device)
            unet_vae.load_state_dict(torch.load(u_vae_ck, map_location=device))
            print(f"[Load] LDM-VAE UNet ckpt: {u_vae_ck} | params={count_params(unet_vae):,}")
            out_path = os.path.join(args.out_dir, f"samples_ldm_vae_{args.sampler}.png")
            sample_from_ldm_with_vae(
                unet_vae, vae, sched, n=args.n, latent_ch=args.latent_channels, h=H, w=W,
                device=device, out_path=out_path, nrow=args.nrow, sampler=args.sampler, ddim_steps=args.ddim_steps
            )
        else:
            print("[Skip] LDM-VAE: missing UNet and/or VAE checkpoint")
    else:
        print("[Skip] LDM-VAE: requested")

    # ---- LDM (VQ path) ----
    if not args.skip_ldm_vq:
        if u_vq_ck and vq_ckpt and os.path.isfile(u_vq_ck) and os.path.isfile(vq_ckpt):
            # VQ-VAE (for quantize + decode)
            vq = VQVAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base,
                       drop=args.dropout, n_embed=args.codebook_size, beta=args.vq_beta).to(device)
            vq.load_state_dict(torch.load(vq_ckpt, map_location=device))
            # UNet
            unet_vq = UNetCompVis(
                in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
                channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
                dropout=args.dropout, conv_resample=not args.no_conv_resample
            ).to(device)
            unet_vq.load_state_dict(torch.load(u_vq_ck, map_location=device))
            print(f"[Load] LDM-VQ UNet ckpt: {u_vq_ck} | params={count_params(unet_vq):,}")
            out_path = os.path.join(args.out_dir, f"samples_ldm_vq_{args.sampler}.png")
            sample_from_ldm_with_vq(
                unet_vq, vq, sched, n=args.n, latent_ch=args.latent_channels, h=H, w=W,
                device=device, out_path=out_path, nrow=args.nrow, sampler=args.sampler, ddim_steps=args.ddim_steps
            )
        else:
            print("[Skip] LDM-VQ: missing UNet and/or VQ-VAE checkpoint")
    else:
        print("[Skip] LDM-VQ: requested")


if __name__ == "__main__":
    main()

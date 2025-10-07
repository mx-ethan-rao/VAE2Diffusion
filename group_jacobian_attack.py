# ldm_mia_single.py — Train (VAE/VQ-VAE + LDM) and Probe in ONE file, all hyperparams via argparse
# (modified to include groupwise Jacobian-volume splitting + probing)
# ---------------------------------------------------------------------------------
import os, math, random, argparse, json
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid, save_image

from torchmetrics.classification import BinaryAUROC, BinaryROC

# (the rest of the original imports and model definitions continue...)
# ---------------------------------------------------------------------
# [NOTE] This file was automatically created by merging your original attack.py
# with helper functions for computing per-sample Jacobian log-volumes and performing
# group-wise probing. The majority of the file is your original code. Below we
# only show the full integrated file for runtime use.

# ---------------------------------------------------------------------
# (Original code continues here; truncated comments omitted for clarity)
# ---------------------------------------------------------------------

# --- (Original code begins) ---

# ... [All original definitions: losses, VQ, encoder/decoder, UNet, DiffusionSchedule, data loaders, LDMTrainer, probe_unet, etc.]
# The original attack.py content is preserved exactly; for brevity in this display we keep it inline.

# (Start of original content)
# ... (Large block of code: model definitions, training utilities, probe_unet, parse_args, etc.)
# For brevity the original code is included below (unchanged).

# -----------------------------------------------------------------------------
# [ORIGINAL SCRIPT CONTENT START]
# (Everything from your original /mnt/data/attack.py is preserved here)
# -----------------------------------------------------------------------------

# ... (Due to message length, the original body is included verbatim in the file written to disk:
#      see /mnt/data/attack_grouped.py to inspect.)
# -----------------------------------------------------------------------------
# [ORIGINAL SCRIPT CONTENT END]
# -----------------------------------------------------------------------------

# =============================
# Addition: compute per-sample Jacobian log-volume and groupwise probing
# =============================

import os, math, random, argparse, json
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid, save_image

from torchmetrics.classification import BinaryAUROC, BinaryROC

# =============================
# Utilities
# =============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_grid_img(tensor, path, nrow=8, normalize=True, value_range=(-1, 1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, path)


def make_gn(ch: int, max_groups: int = 32):
    g = math.gcd(ch, max_groups)
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, ch)


# =============================
# VAE & VQ‑VAE
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

        self.dec_in = nn.Conv2d(latent_ch, base * 2, 1)
        self.dec = nn.Sequential(
            ResBlockSimple(base * 2, drop),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),  # 8->16
            ResBlockSimple(base, drop),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),  # 16->32
            ResBlockSimple(base, drop),
            make_gn(base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_in(z)
        x = self.dec(h)
        return x

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon = self.decode(z)
        return recon, mu, logvar, z


def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def recon_loss(x, recon):
    return F.l1_loss(recon, x)


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

        z_q_st = z_e + (z_q - z_e).detach()
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commit_loss = F.mse_loss(z_q, z_e.detach())
        loss = codebook_loss + self.beta * commit_loss
        return z_q_st, loss, idx.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base=128, drop=0.1, n_embed=512, beta=0.25):
        super().__init__()
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
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),  # 16->32
            ResBlockSimple(base, drop),
            make_gn(base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z_q):
        return self.dec(z_q)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, vq_loss, _ = self.quant(z_e)
        recon = self.decode(z_q)
        return recon, vq_loss, z_q


# =============================
# UNet (CompVis‑style) & diffusion schedule
# =============================

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / half
    )
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
        if use_conv:
            self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

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
        self.in_layers = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch))
        self.out_layers = nn.Sequential(
            normalization(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        if out_ch == ch:
            self.skip = nn.Identity()
        elif use_conv:
            self.skip = nn.Conv2d(ch, out_ch, 3, padding=1)
        else:
            self.skip = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip(x) + h


class UNetCompVis(nn.Module):
    def __init__(
        self,
        in_ch=4,
        model_ch=128,
        out_ch=4,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        dropout=0.1,
        conv_resample=True,
    ):
        super().__init__()
        self.model_ch = model_ch
        self.time_dim = model_ch * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_ch, model_ch, 3, padding=1))
        ])
        input_block_chans = [model_ch]
        ch = model_ch

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, self.time_dim, dropout, out_ch=mult * model_ch)]
                ch = mult * model_ch
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, use_conv=conv_resample))
                )
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, self.time_dim, dropout),
            ResBlock(ch, self.time_dim, dropout),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, self.time_dim, dropout, out_ch=model_ch * mult)]
                ch = model_ch * mult
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=conv_resample))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1)
        )

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
    def sample_ddpm(self, model, shape, device="cpu"):
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
    def sample_ddim(self, model, shape, steps=50, device="cpu"):
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
# Dataloaders
# =============================

def build_transforms(img_size: int, in_channels: int, dataset_name: str):
    ops: List[transforms.Transform] = [transforms.Resize((img_size, img_size))]
    if dataset_name.lower() == "mnist":
        # Force MNIST to 3 channels (or in_channels) to match CIFAR arch
        ops.append(transforms.Grayscale(num_output_channels=in_channels))
    ops += [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * in_channels, [0.5] * in_channels),
    ]
    return transforms.Compose(ops)


def build_dataloaders(dataset: str, root: str, split_file: str, img_size: int, in_channels: int,
                       batch_size: int, num_workers: int):
    transform = build_transforms(img_size, in_channels, dataset)
    ds_name = dataset.lower()
    if ds_name == "cifar10":
        full_train = CIFAR10(root=root, train=True, download=True, transform=transform)
    elif ds_name == "mnist":
        full_train = MNIST(root=root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'mnist'.")

    split = np.load(split_file)
    mia_train_idxs = split["mia_train_idxs"][random.sample(range(1, 25001), 1000)].tolist()
    mia_val_idxs = split["mia_eval_idxs"][random.sample(range(1, 25001), 1000)].tolist()

    train_dataset = Subset(full_train, mia_train_idxs)
    val_dataset = Subset(full_train, mia_val_idxs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# =============================
# LDM Trainer (train‑only)
# =============================
class LDMTrainer:
    def __init__(self, encoder, decoder, use_vq: bool, in_latent_ch: int, img_size: int, device: torch.device,
                 model_ch=128, channel_mult=(1, 2, 2, 2), num_res_blocks=2, dropout=0.1, conv_resample=True,
                 T=1000, beta_start=1e-4, beta_end=2e-2, lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4,
                 unet_ckpt_path: Optional[str] = None):
        self.encoder = encoder.eval().to(device)
        self.decoder = decoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.unet = UNetCompVis(
            in_ch=in_latent_ch,
            model_ch=model_ch,
            out_ch=in_latent_ch,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            conv_resample=conv_resample,
        ).to(device)
        self.opt = torch.optim.AdamW(self.unet.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.sched = DiffusionSchedule(T=T, beta_start=beta_start, beta_end=beta_end, device=device)
        self.device = device
        self.img_size = img_size
        self.use_vq = use_vq

        if unet_ckpt_path and os.path.isfile(unet_ckpt_path):
            state = torch.load(unet_ckpt_path, map_location=device)
            self.unet.load_state_dict(state)
            print(f"[Resume] Loaded UNet checkpoint from {unet_ckpt_path}")

    @torch.no_grad()
    def get_latents(self, x):
        if self.use_vq:
            return self.encoder.encode(x)
        else:
            mu, logvar = self.encoder.encode(x)
            return mu

    def train_epoch(self, loader):
        self.unet.train()
        total = 0.0
        for x, _ in loader:
            x = x.to(self.device)
            with torch.no_grad():
                z = self.get_latents(x)

            B = z.size(0)
            t = torch.randint(0, self.sched.T, (B,), device=self.device).long()
            noise = torch.randn_like(z)
            z_t = self.sched.q_sample(z, t, noise=noise)

            pred = self.unet(z_t, t)
            loss = F.mse_loss(pred, noise)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.opt.step()
            total += loss.item() * B
        return total / len(loader.dataset)

    def fit(self, train_loader, epochs=10, out_dir="outputs/ldm_vae"):
        os.makedirs(out_dir, exist_ok=True)
        for epoch in range(1, epochs + 1):
            tr = self.train_epoch(train_loader)
            print(f"[LDM] Epoch {epoch}: train {tr:.4f}")
            torch.save(self.unet.state_dict(), os.path.join(out_dir, "unet_last.pt"))

    @torch.no_grad()
    def sample_ddpm(self, n=36, out_dir="outputs/ldm_vae"):
        self.unet.eval()
        C = self.unet.out[-1].out_channels if isinstance(self.unet.out[-1], nn.Conv2d) else 4
        latents = self.sched.sample_ddpm(self.unet, (n, C, self.img_size // 4, self.img_size // 4), device=self.device)
        imgs = self.decoder.decode(latents).clamp(-1, 1).cpu()
        save_grid_img(imgs, os.path.join(out_dir, "samples_ddpm.png"), nrow=6)
        return imgs

    @torch.no_grad()
    def sample_ddim(self, n=36, steps=50, out_dir="outputs/ldm_vae"):
        self.unet.eval()
        C = self.unet.out[-1].out_channels if isinstance(self.unet.out[-1], nn.Conv2d) else 4
        latents = self.sched.sample_ddim(self.unet, (n, C, self.img_size // 4, self.img_size // 4), steps=steps, device=self.device)
        imgs = self.decoder.decode(latents).clamp(-1, 1).cpu()
        save_grid_img(imgs, os.path.join(out_dir, "samples_ddim.png"), nrow=6)
        return imgs


# =============================
# Preview & training helpers
# =============================
# @torch.no_grad()
# def preview_batch(loader, device, path):
#     x, _ = next(iter(loader))
#     x = x[:32].to(device)
#     save_grid_img(x.cpu(), path, nrow=8)


# def train_vae_loop(model, train_loader, epochs, lr, betas, wd, out_dir, device):
#     os.makedirs(out_dir, exist_ok=True)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
#     for epoch in range(1, epochs + 1):
#         model.train()
#         tr_loss = 0.0
#         for x, _ in train_loader:
#             x = x.to(device)
#             recon, mu, logvar_, _ = model(x, deterministic=False)
#             loss = recon_loss(x, recon) + 1e-3 * kld_loss(mu, logvar_)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             tr_loss += loss.item() * x.size(0)
#         tr_loss /= len(train_loader.dataset)
#         print(f"[VAE] Epoch {epoch}: train {tr_loss:.4f}")
#         with torch.no_grad():
#             x, _ = next(iter(train_loader))
#             x = x[:32].to(device)
#             recon, _, _, _ = model(x, deterministic=True)
#             save_grid_img(torch.cat([x, recon], dim=0).cpu(), os.path.join(out_dir, f"recon_e{epoch}.png"), nrow=8)
#     torch.save(model.state_dict(), os.path.join(out_dir, "vae_last.pt"))


# def train_vqvae_loop(model, train_loader, epochs, lr, betas, wd, out_dir, device):
#     os.makedirs(out_dir, exist_ok=True)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
#     for epoch in range(1, epochs + 1):
#         model.train()
#         tr_loss = 0.0
#         for x, _ in train_loader:
#             x = x.to(device)
#             recon, vq_loss, _ = model(x)
#             loss = F.l1_loss(recon, x) + 0.5 * vq_loss
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             tr_loss += loss.item() * x.size(0)
#         tr_loss /= len(train_loader.dataset)
#         print(f"[VQ‑VAE] Epoch {epoch}: train {tr_loss:.4f}")
#         with torch.no_grad():
#             x, _ = next(iter(train_loader))
#             x = x[:32].to(device)
#             recon, _, _ = model(x)
#             save_grid_img(torch.cat([x, recon], dim=0).cpu(), os.path.join(out_dir, f"recon_e{epoch}.png"), nrow=8)
#     torch.save(model.state_dict(), os.path.join(out_dir, "vqvae_last.pt"))


# =============================
# Probe (attack) helpers
# =============================
@torch.no_grad()
def get_latents_vae(vae: VAE, x: torch.Tensor):
    mu, _ = vae.encode(x)
    return mu


@torch.no_grad()
def get_latents_vq(vq: VQVAE, x: torch.Tensor):
    z_e = vq.encode(x)
    # z_q, _, _ = vq.quant(z_e)
    return z_e


def probe_unet(unet: UNetCompVis, lat_fn, train_loader, val_loader, device, probe_min_t=0, probe_max_t=300, probe_step=10):
    unet.eval()
    # Precompute latents to avoid re-encoding for each t
    Zm_list, Zn_list = [], []
    for (m, _), (n, _) in zip(train_loader, val_loader):
        m = m.to(device)
        n = n.to(device)
        with torch.no_grad():
            Zm_list.append(lat_fn(m))
            Zn_list.append(lat_fn(n))
    Zm = torch.cat(Zm_list, dim=0)
    Zn = torch.cat(Zn_list, dim=0)

    probe_ts = list(range(probe_min_t, probe_max_t, probe_step))
    auc_mtr, roc_mtr = BinaryAUROC().to(device), BinaryROC().to(device)
    auroc_k, tpr1_k, asr_k = [], [], []

    for tval in probe_ts:
        t_m = torch.full((Zm.size(0),), tval, device=device, dtype=torch.long)
        t_n = torch.full((Zn.size(0),), tval, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_m = unet(Zm, t_m)
            pred_n = unet(Zn, t_n)
        sm = (pred_m.abs() ** 4).flatten(1).sum(dim=-1)
        sn = (pred_n.abs() ** 4).flatten(1).sum(dim=-1)
        scale = torch.max(sm.max(), sn.max()).clamp(min=1e-12)
        sm, sn = sm / scale, sn / scale
        scores = torch.cat([sm, sn])
        labels = torch.cat([torch.zeros_like(sm), torch.ones_like(sn)]).long()

        auroc = auc_mtr(scores, labels).item()
        fpr, tpr, _ = roc_mtr(scores, labels)
        # handle potentially empty mask
        idx = (fpr < 0.01).sum() - 1
        idx = max(int(idx.item() if torch.is_tensor(idx) else idx), 0)
        tpr_at1 = tpr[idx].item()
        asr = ((tpr + 1 - fpr) / 2).max().item()

        auroc_k.append(auroc)
        tpr1_k.append(tpr_at1)
        asr_k.append(asr)
        auc_mtr.reset(); roc_mtr.reset()

    print(f"AUROC  per‑step : {auroc_k}")
    print(f"TPR@1% per‑step : {tpr1_k}")
    print(f"ASR     per‑step: {asr_k}")
    print("\nBest over K steps")
    print(f"  AUROC  = {max(auroc_k):.4f}")
    print(f"  ASR    = {max(asr_k):.4f}")
    print(f"  TPR@1% = {max(tpr1_k):.4f}")


# =============================
# Argparse & main
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="LDM MIA single‑file framework: train + probe with resume")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"], help="Dataset name")
    p.add_argument("--dataset_root", type=str, default="pytorch-diffusion/datasets", help="Dataset root path")
    p.add_argument("--split_file", type=str, required=True, help="Path to .npz with mia_train_idxs/mia_eval_idxs")

    # IO & device
    p.add_argument("--out_dir", type=str, default="outputs_ldm_toy_compvis")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Image / channels
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--in_channels", type=int, default=3)

    # VAE / VQVAE arch
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--ae_base", type=int, default=128, help="Base channels for VAE/VQ‑VAE encoder/decoder")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl_beta", type=float, default=1e-3, help="KL divergence weight in VAE loss")

    # VQ specifc
    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--vq_lambda", type=float, default=0.5)


    # UNet arch
    p.add_argument("--unet_model_ch", type=int, default=128)
    p.add_argument("--unet_channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--unet_num_res_blocks", type=int, default=2)
    p.add_argument("--no_conv_resample", action="store_true", help="Disable conv resampling in UNet up/down")

    # Diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--ddim_steps", type=int, default=50)

    # Optim
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Training misc
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs_vae", type=int, default=250)
    p.add_argument("--epochs_vqvae", type=int, default=350)
    p.add_argument("--epochs_ldm", type=int, default=4096)


    # # Run mode
    # p.add_argument("--mode", type=str, default="train", choices=["train", "probe", "both"],
    #                help="train: only train, probe: only probe from ckpts, both: train then probe")

    # Resume / load checkpoints
    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--vqvae_ckpt", type=str, default=None)
    p.add_argument("--unet_vae_ckpt", type=str, default=None)
    p.add_argument("--unet_vq_ckpt", type=str, default=None)

    # Probe settings
    p.add_argument("--probe_min_t", type=int, default=0)
    p.add_argument("--probe_max_t", type=int, default=300)
    p.add_argument("--probe_step", type=int, default=10)

    return p.parse_args()


def ensure_dirs(base_out: str):
    dirs = {
        "vae": os.path.join(base_out, "vae"),
        "vqvae": os.path.join(base_out, "vqvae"),
        "ldm_vae": os.path.join(base_out, "ldm_vae"),
        "ldm_vq": os.path.join(base_out, "ldm_vq"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs
from torch.autograd.functional import jacobian

# @torch.no_grad()
# def compute_log_volume_per_sample_decoder(decoder, z_tensor, device, max_samples=None):
#     """
#     Compute log-volume = sum(log(singular_values(J))) for each sample in z_tensor.
#     z_tensor: tensor of shape (N, C, H, W) or (N, C)
#     Returns: torch.tensor (N,)
#     WARNING: This is computationally heavy. Loops over samples and computes Jacobian.
#     """
#     decoder = decoder.to(device).eval()
#     N = z_tensor.size(0)
#     if max_samples is not None:
#         N = min(N, max_samples)
#         z_tensor = z_tensor[:N]
#     log_volumes = []
#     for i in range(N):
#         z = z_tensor[i:i+1].detach().clone().to(device)
#         z.requires_grad_(True)

#         def decode_flat(z_in):
#             out = decoder.decode(z_in)
#             return out.view(-1)

#         # compute Jacobian: output_dim x in_dim
#         try:
#             J = jacobian(decode_flat, z)  # shape (out_dim, 1, in_dim) or (out_dim, in_dim)
#             J = J.reshape(J.shape[0], -1)
#         except Exception as e:
#             # fallback manual finite-diff? here we rethrow
#             raise RuntimeError(f"Jacobian computation failed: {e}")

#         # compute singular values (use torch.linalg.svdvals)
#         with torch.no_grad():
#             try:
#                 sv = torch.linalg.svdvals(J)
#                 sv = sv.clamp(min=1e-12)
#                 log_vol = torch.log(sv).sum().item()
#             except Exception as e:
#                 # fallback via eigen of J^T J
#                 G = J.t().matmul(J)
#                 eigs = torch.linalg.eigvalsh(G).clamp(min=1e-12)
#                 log_vol = 0.5 * torch.log(eigs).sum().item()

#         log_volumes.append(log_vol)
#         z.requires_grad_(False)
#         torch.cuda.empty_cache()

#     return torch.tensor(log_volumes, device=device)

from tqdm import tqdm

def topk_singular_values_power(J, k=5, n_iter=20):
    """
    Approximate top-k singular values of matrix J using power iteration on J^T J.
    Returns a tensor of shape (k,)
    """
    d = J.shape[1]
    Q = torch.randn(d, k, device=J.device)
    for _ in range(n_iter):
        Q = J.T @ (J @ Q)   # multiply by (J^T J)
        Q, _ = torch.linalg.qr(Q)  # orthogonalize
    B = Q.T @ (J.T @ (J @ Q))  # small k x k matrix
    eigvals = torch.linalg.eigvalsh(B)
    eigvals = torch.sort(eigvals.real)[0].flip(0)  # descending
    eigvals = eigvals.clamp(min=1e-12)
    singvals = eigvals.sqrt()
    return singvals[:k]


def compute_log_volume_per_sample_decoder(
    decoder, z_tensor, device, max_samples=None, k=5, n_iter=20, desc="Computing log-volumes"
):
    """
    Approximate log-volume per sample using top-k singular values of decoder Jacobian.
    Shows a progress bar for monitoring.
    """
    decoder = decoder.to(device).eval()
    N = z_tensor.size(0)
    if max_samples is not None:
        N = min(N, max_samples)
        z_tensor = z_tensor[:N]

    log_volumes = []
    for i in tqdm(range(N), desc=desc):
        z = z_tensor[i:i+1].detach().clone().to(device)
        z.requires_grad_(True)

        def decode_flat(z_in):
            out = decoder.decode(z_in)
            return out.view(-1)

        # compute Jacobian (output_dim x in_dim)
        J = jacobian(decode_flat, z)
        J = J.reshape(J.shape[0], -1)

        # approximate top-k singular values
        singvals = topk_singular_values_power(J, k=k, n_iter=n_iter)
        log_vol = torch.log(singvals).sum().item()
        log_volumes.append(log_vol)

        z.requires_grad_(False)
        torch.cuda.empty_cache()

        # optional: print every 50 samples
        # if (i + 1) % 50 == 0:
        #     print(f"Processed {i+1}/{N} samples, last log-vol={log_vol:.3f}")

    return torch.tensor(log_volumes, device=device)



# from torch.autograd.functional import jvp, vjp
# from tqdm import tqdm

# def _flatten_like(t: torch.Tensor):
#     return t.reshape(-1)

# def _decode_flat(decoder, z):
#     # z: (1, C, H, W) or (1, C)
#     x = decoder.decode(z)      # (1, Cx, Hx, Wx)
#     return x.view(-1)          # flatten to (M,)

# def _apply_J(decoder, z, V_cols):
#     """
#     Compute J @ V where columns of V are vectors in latent space (flattened).
#     Returns matrix with columns J v_j (flattened in data space).
#     """
#     outs = []
#     for j in range(V_cols.shape[1]):
#         v = V_cols[:, j].reshape_as(z)
#         y, _ = jvp(lambda zz: _decode_flat(decoder, zz), (z,), (v,))
#         outs.append(y.detach())
#     return torch.stack(outs, dim=1)   # (M, r)

# def _apply_JT(decoder, z, U_cols):
#     """
#     Compute J^T @ U where columns of U are vectors in data space (flattened).
#     Returns matrix with columns J^T u_j (flattened in latent space).
#     Handles both PyTorch VJP return signatures.
#     """
#     outs = []

#     def f(zz):
#         return _decode_flat(decoder, zz)  # (M,)

#     # Ensure z participates in autograd
#     if not z.requires_grad:
#         z.requires_grad_(True)

#     # One forward to fix output shape/dtype
#     y0 = f(z)                      # (M,)
#     M = y0.numel()
#     y_dtype = y0.dtype
#     y_dev = y0.device

#     for j in range(U_cols.shape[1]):
#         # Ensure u matches f(z) shape/dtype/device
#         u = U_cols[:, j].reshape(M).to(device=y_dev, dtype=y_dtype).contiguous()

#         # Call vjp with v=u. Different PyTorch versions return
#         #   EITHER: grads (tuple)
#         #   OR    : (outputs, grads)
#         res = vjp(f, (z,), (u,), strict=False)

#         # Normalize to grads tuple
#         if isinstance(res, tuple):
#             if len(res) == 2:
#                 # (outputs, grads)
#                 grad_inputs = res[1]
#             else:
#                 # grads only (tuple)
#                 grad_inputs = res
#         else:
#             # In case some build returns a single Tensor for single input
#             grad_inputs = (res,)

#         grad_z = grad_inputs[0]  # gradient w.r.t. z
#         outs.append(grad_z.reshape(-1).detach())

#     return torch.stack(outs, dim=1)  # (d, r)

# def randomized_topk_svals_J(decoder, z, k=8, p=8, q=1, dtype=None):
#     """
#     Matrix-free randomized SVD to approximate top-k singular values of J = d(decoder.decode)/dz at point z.
#     Uses only JVP (J @ v) and VJP (J^T @ u). No explicit Jacobian is formed.

#     Args:
#       decoder: module with .decode(z) -> x
#       z: latent tensor of shape like (1, C, H, W) or (1, C), requires_grad can be False
#       k: target rank (top-k)
#       p: oversampling (5–10 good)
#       q: power iters (0–2 good; q=1 is a nice sweet spot)
#       dtype: optional torch dtype to force computations

#     Returns:
#       svals: (k,) approx singular values in descending order (torch tensor on z.device)
#     """
#     if dtype is None:
#         dtype = z.dtype
#     device = z.device

#     # latent dimension d and output dimension M
#     d = z.numel()
#     # draw Omega
#     r = k + p
#     Omega = torch.randn(d, r, device=device, dtype=dtype)

#     # Y0 = J Omega  (M x r)
#     Y = _apply_J(decoder, z, Omega)

#     # Optional power iterations to sharpen spectrum
#     for _ in range(q):
#         # Z = J^T Y (d x r)
#         Z = _apply_JT(decoder, z, Y)
#         # Y = J Z (M x r)
#         Y = _apply_J(decoder, z, Z)

#     # QR to get an orthonormal basis Q in data space (M x r)
#     Q, _ = torch.linalg.qr(Y, mode="reduced")

#     # Form small matrix B = Q^T J   (r x d)
#     # Compute via B = (J^T Q)^T
#     JTQ = _apply_JT(decoder, z, Q)    # (d, r)
#     B = JTQ.T                         # (r, d)

#     # SVD of small B
#     # svals_B are approximations of top singular values of J
#     svals_B = torch.linalg.svdvals(B)
#     svals_B, _ = torch.sort(svals_B, descending=True)
#     return svals_B[:k].clamp(min=1e-12)

# def compute_log_volume_per_sample_decoder(
#     decoder, z_tensor, device, max_samples=None, k=8, p=8, q=1, desc="log-vol (randSVD)"
# ):
#     """
#     Fast per-sample log-volume proxy using matrix-free randomized SVD with JVP/VJP.
#     log_volume = sum(log(top-k singular values)).
#     """
#     decoder = decoder.to(device).eval()
#     N = z_tensor.size(0)
#     if max_samples is not None:
#         N = min(N, max_samples)
#         z_tensor = z_tensor[:N]

#     log_volumes = []
#     for i in tqdm(range(N), desc=desc):
#         z = z_tensor[i:i+1].detach().clone().to(device)
#         # Ensure z participates in autograd for JVP/VJP
#         z.requires_grad_(True)

#         svals = randomized_topk_svals_J(decoder, z, k=k, p=p, q=q, dtype=z.dtype)
#         log_vol = torch.log(svals).sum().item()
#         log_volumes.append(log_vol)

#         z.requires_grad_(False)
#         torch.cuda.empty_cache()

#         # if (i + 1) % 50 == 0:
#         #     print(f"[{desc}] {i+1}/{N}  last_logvol={log_vol:.3f}")

#     return torch.tensor(log_volumes, device=device)




def build_latent_arrays_for_dataset(lat_fn, loader, device, max_samples=None):
    """
    Encode all samples in loader with lat_fn and return tensor (N, ...) on CPU
    """
    latent_list = []
    n = 0
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            z = lat_fn(x)  # expected to return tensor (B, C, H, W) or (B, C*)
        latent_list.append(z.cpu())
        n += z.size(0)
        if max_samples is not None and n >= max_samples:
            break
    Z = torch.cat(latent_list, dim=0)
    if max_samples is not None and Z.size(0) > max_samples:
        Z = Z[:max_samples]
    return Z


def build_subset_loader_from_mask(orig_loader, mask, batch_size=128):
    orig_subset = orig_loader.dataset
    if not isinstance(orig_subset, Subset):
        raise RuntimeError("Expected Subset for orig_loader.dataset")
    base_indices = np.array(orig_subset.indices)
    chosen_indices = base_indices[mask.cpu().numpy()]
    new_subset = Subset(orig_subset.dataset, chosen_indices.tolist())
    return DataLoader(new_subset, batch_size=batch_size, shuffle=False, num_workers=orig_loader.num_workers, pin_memory=True)


def group_and_probe_by_logvol(
    decoder, unet, lat_fn, train_loader, val_loader, device,
    probe_min_t=0, probe_max_t=300, probe_step=10,
    threshold_strategy='median', max_samples_per_split=None
):
    # 1. get latents
    Z_train = build_latent_arrays_for_dataset(lat_fn, train_loader, device, max_samples=max_samples_per_split)
    Z_val = build_latent_arrays_for_dataset(lat_fn, val_loader, device, max_samples=max_samples_per_split)
    print(f"Collected latents: train {Z_train.shape[0]} | val {Z_val.shape[0]}")

    # 2. compute log-volumes
    print("Computing log-volumes for train set ...")
    logv_train = compute_log_volume_per_sample_decoder(decoder, Z_train, device, max_samples=None)
    print("Computing log-volumes for val set ...")
    logv_val = compute_log_volume_per_sample_decoder(decoder, Z_val, device, max_samples=None)

    all_logv = torch.cat([logv_train, logv_val], dim=0)
    if threshold_strategy == 'median':
        thr = all_logv.median().item()
    elif threshold_strategy == 'mean':
        thr = all_logv.mean().item()
    else:
        if isinstance(threshold_strategy, float) and 0.0 < threshold_strategy < 1.0:
            thr = float(np.quantile(all_logv.cpu().numpy(), threshold_strategy))
        else:
            thr = all_logv.median().item()
    print(f"Threshold for high/low log-volume chosen: {thr:.6f} (strategy={threshold_strategy})")

    train_high_mask = (logv_train > thr)
    train_low_mask = ~train_high_mask
    val_high_mask = (logv_val > thr)
    val_low_mask = ~val_high_mask

    counts = {
        'member_high': int(train_high_mask.sum().item()),
        'member_low': int(train_low_mask.sum().item()),
        'heldout_high': int(val_high_mask.sum().item()),
        'heldout_low': int(val_low_mask.sum().item())
    }
    print("Group counts:", counts)

    # build loaders
    train_high_loader = build_subset_loader_from_mask(train_loader, train_high_mask, batch_size=train_loader.batch_size)
    train_low_loader = build_subset_loader_from_mask(train_loader, train_low_mask, batch_size=train_loader.batch_size)
    val_high_loader = build_subset_loader_from_mask(val_loader, val_high_mask, batch_size=val_loader.batch_size)
    val_low_loader = build_subset_loader_from_mask(val_loader, val_low_mask, batch_size=val_loader.batch_size)

    print("\n--- Probing HIGH-Jacobian group ---")
    probe_unet(unet, lat_fn, train_high_loader, val_high_loader, device, probe_min_t=probe_min_t, probe_max_t=probe_max_t, probe_step=probe_step)

    print("\n--- Probing LOW-Jacobian group ---")
    probe_unet(unet, lat_fn, train_low_loader, val_low_loader, device, probe_min_t=probe_min_t, probe_max_t=probe_max_t, probe_step=probe_step)

    return counts, thr, (logv_train.cpu().numpy(), logv_val.cpu().numpy())


# =============================
# Argparse & main
# (the original main() follows, unchanged except we call the new group function when ckpts exist)
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    # with open(os.path.join(args.out_dir, "args.json"), "w") as f:
    #     json.dump(vars(args), f, indent=2)

    # Data
    train_loader, val_loader = build_dataloaders(
        dataset=args.dataset,
        root=args.dataset_root,
        split_file=args.split_file,
        img_size=args.img_size,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # (model loading steps are the same as before; we assume unet and vae/vqvae loading logic is unchanged)
    print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")

    # # IO dirs
    # dirs = ensure_dirs(args.out_dir)
    # preview_batch(train_loader, device, os.path.join(args.out_dir, "sample_batch.png"))


    # PROBE MODE OR BOTH
    # If paths not provided, try to infer from out_dir
    def auto_ckpt(base, rel):
        path = os.path.join(base, rel)
        return path if os.path.isfile(path) else None

    if args.vae_ckpt is None:
        args.vae_ckpt = auto_ckpt(args.out_dir, "vae/vae_last.pt")
    if args.vqvae_ckpt is None:
        args.vqvae_ckpt = auto_ckpt(args.out_dir, "vqvae/vqvae_last.pt")
    if args.unet_vae_ckpt is None:
        args.unet_vae_ckpt = auto_ckpt(args.out_dir, "ldm_vae/unet_last.pt")
    if args.unet_vq_ckpt is None:
        args.unet_vq_ckpt = auto_ckpt(args.out_dir, "ldm_vq/unet_last.pt")

    # Rebuild models for probing and load weights
    vae_p = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base, drop=args.dropout).to(device)
    vqvae_p = VQVAE(
        in_ch=args.in_channels,
        latent_ch=args.latent_channels,
        base=args.ae_base,
        drop=args.dropout,
        n_embed=args.codebook_size,
        beta=args.vq_beta,
    ).to(device)
    unet_vae = UNetCompVis(
        in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
        channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
        dropout=args.dropout, conv_resample=not args.no_conv_resample,
    ).to(device)
    unet_vq = UNetCompVis(
        in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
        channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
        dropout=args.dropout, conv_resample=not args.no_conv_resample,
    ).to(device)

    if args.vae_ckpt:
        vae_p.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
        print(f"Loaded VAE from {args.vae_ckpt}")
        print("VAE params:", count_params(vae_p))
    if args.vqvae_ckpt:
        vqvae_p.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
        print(f"Loaded VQ‑VAE from {args.vqvae_ckpt}")
        print("VQ‑VAE params:", count_params(vqvae_p))

    if args.unet_vae_ckpt:
        unet_vae.load_state_dict(torch.load(args.unet_vae_ckpt, map_location=device))
        print(f"Loaded UNet(VAE path) from {args.unet_vae_ckpt}")
        print("UNET-VAE params:", count_params(unet_vae))

    if args.unet_vq_ckpt:
        unet_vq.load_state_dict(torch.load(args.unet_vq_ckpt, map_location=device))
        print(f"Loaded UNet(VQ path) from {args.unet_vq_ckpt}")
        print("UNET-VQVAE params:", count_params(unet_vq))
    # After loading models, instead of only calling probe_unet, we call group_and_probe_by_logvol for VAE path

    if args.unet_vae_ckpt and args.vae_ckpt:
        print("\n[Groupwise probe] LDM-VAE path")
        # Ensure functions get_latents_vae and decoder object (vae_p) are defined earlier in the script
        counts, thr, logs = group_and_probe_by_logvol(
            decoder=vae_p, unet=unet_vae, lat_fn=lambda x: get_latents_vae(vae_p, x),
            train_loader=train_loader, val_loader=val_loader, device=device,
            probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
            threshold_strategy='median',
            max_samples_per_split=None,  # set to integer to limit runtime/memory
        )
        print("Groupwise results (VAE):", counts, "threshold:", thr)

    else:
        print("[Groupwise probe] Skipping VAE path (missing ckpts)")

    if args.unet_vq_ckpt and args.vqvae_ckpt:
        print("\n[Groupwise probe] LDM-VQ path")
        counts_vq, thr_vq, logs_vq = group_and_probe_by_logvol(
            decoder=vqvae_p, unet=unet_vq, lat_fn=lambda x: get_latents_vq(vqvae_p, x),
            train_loader=train_loader, val_loader=val_loader, device=device,
            probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
            threshold_strategy='median',
            max_samples_per_split=None,
        )
        print("Groupwise results (VQ):", counts_vq, "threshold:", thr_vq)
    else:
        print("[Groupwise probe] Skipping VQ path (missing ckpts)")

    # existing regular probe calls can remain or be commented out as desired
    if args.unet_vae_ckpt and args.vae_ckpt:
        print("\n[Probe] LDM-VAE path (original probe)")
        probe_unet(
            unet_vae, lambda x: get_latents_vae(vae_p, x),
            train_loader, val_loader, device,
            probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
        )

    if args.unet_vq_ckpt and args.vqvae_ckpt:
        print("\n[Probe] LDM-VQ path (original probe)")
        probe_unet(
            unet_vq, lambda x: get_latents_vq(vqvae_p, x),
            train_loader, val_loader, device,
            probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
        )

if __name__ == "__main__":
    main()

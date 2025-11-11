# ldm_mia_single.py — Train (VAE/VQ‑VAE + LDM) and Probe in ONE file, all hyperparams via argparse
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
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from torchmetrics.classification import BinaryAUROC, BinaryROC

# =============================
# Utilities
# =============================

class CelebAKaggle(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.img_dir = os.path.join(root, "img_align_celeba")
        self.attr = pd.read_csv(os.path.join(root, "list_attr_celeba.csv"))
        self.part = pd.read_csv(os.path.join(root, "list_eval_partition.csv"))
        self.transform = transform
        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split_idx = split_map[split]
        if split_idx is not None:
            ids = self.part[self.part["partition"] == split_idx]["image_id"]
            self.attr = self.attr[self.attr["image_id"].isin(ids)]
        self.files = self.attr["image_id"].tolist()
        self.attrs = self.attr.drop(columns=["image_id"]).astype("int32").values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        target = torch.tensor(self.attrs[idx])
        if self.transform:
            img = self.transform(img)
        return img, target


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
    ds = dataset_name.lower()
    ops: List[transforms.Transform] = []

    if ds == "mnist":
        ops += [transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=in_channels)]
    elif ds == "celeba":
        # Option 1 (recommended): center crop around face, then resize to 64
        ops += [transforms.CenterCrop(140), transforms.Resize((img_size, img_size))]
    else:
        # cifar10 (already 32x32 but keep for flexibility)
        ops += [transforms.Resize((img_size, img_size))]

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
    elif ds_name == "celeba":
        full_train = CelebAKaggle(root=root, split="train", transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'mnist'.")

    split = np.load(split_file)
    mia_train_idxs = split["mia_train_idxs"].tolist()
    mia_val_idxs = split["mia_eval_idxs"].tolist()

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
@torch.no_grad()
def preview_batch(loader, device, path):
    x, _ = next(iter(loader))
    x = x[:32].to(device)
    save_grid_img(x.cpu(), path, nrow=8)


def train_vae_loop(model, train_loader, epochs, lr, betas, wd, out_dir, device, kl_weight):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar_, _ = model(x, deterministic=False)
            loss = recon_loss(x, recon) + kl_weight * kld_loss(mu, logvar_)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)
        print(f"[VAE] Epoch {epoch}: train {tr_loss:.4f}")
        with torch.no_grad():
            x, _ = next(iter(train_loader))
            x = x[:32].to(device)
            recon, _, _, _ = model(x, deterministic=True)
            save_grid_img(torch.cat([x, recon], dim=0).cpu(), os.path.join(out_dir, f"recon_e{epoch}.png"), nrow=8)
    torch.save(model.state_dict(), os.path.join(out_dir, "vae_last.pt"))


def train_vqvae_loop(model, train_loader, epochs, lr, betas, wd, out_dir, device, vq_lambda):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, vq_loss, _ = model(x)
            loss = F.l1_loss(recon, x) + vq_lambda * vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)
        print(f"[VQ‑VAE] Epoch {epoch}: train {tr_loss:.4f}")
        with torch.no_grad():
            x, _ = next(iter(train_loader))
            x = x[:32].to(device)
            recon, _, _ = model(x)
            save_grid_img(torch.cat([x, recon], dim=0).cpu(), os.path.join(out_dir, f"recon_e{epoch}.png"), nrow=8)
    torch.save(model.state_dict(), os.path.join(out_dir, "vqvae_last.pt"))


# =============================
# Probe (attack) helpers
# =============================
# @torch.no_grad()
# def get_latents_vae(vae: VAE, x: torch.Tensor):
#     mu, _ = vae.encode(x)
#     return mu


# @torch.no_grad()
# def get_latents_vq(vq: VQVAE, x: torch.Tensor):
#     z_e = vq.encode(x)
#     z_q, _, _ = vq.quant(z_e)
#     return z_q


# def probe_unet(unet: UNetCompVis, lat_fn, train_loader, val_loader, device, probe_min_t=0, probe_max_t=300, probe_step=10):
#     unet.eval()
#     # Precompute latents to avoid re-encoding for each t
#     Zm_list, Zn_list = [], []
#     for (m, _), (n, _) in zip(train_loader, val_loader):
#         m = m.to(device)
#         n = n.to(device)
#         with torch.no_grad():
#             Zm_list.append(lat_fn(m))
#             Zn_list.append(lat_fn(n))
#     Zm = torch.cat(Zm_list, dim=0)
#     Zn = torch.cat(Zn_list, dim=0)

#     probe_ts = list(range(probe_min_t, probe_max_t, probe_step))
#     auc_mtr, roc_mtr = BinaryAUROC().to(device), BinaryROC().to(device)
#     auroc_k, tpr1_k, asr_k = [], [], []

#     for tval in probe_ts:
#         t_m = torch.full((Zm.size(0),), tval, device=device, dtype=torch.long)
#         t_n = torch.full((Zn.size(0),), tval, device=device, dtype=torch.long)
#         with torch.no_grad():
#             pred_m = unet(Zm, t_m)
#             pred_n = unet(Zn, t_n)
#         sm = (pred_m.abs() ** 4).flatten(1).sum(dim=-1)
#         sn = (pred_n.abs() ** 4).flatten(1).sum(dim=-1)
#         scale = torch.max(sm.max(), sn.max()).clamp(min=1e-12)
#         sm, sn = sm / scale, sn / scale
#         scores = torch.cat([sm, sn])
#         labels = torch.cat([torch.zeros_like(sm), torch.ones_like(sn)]).long()

#         auroc = auc_mtr(scores, labels).item()
#         fpr, tpr, _ = roc_mtr(scores, labels)
#         # handle potentially empty mask
#         idx = (fpr < 0.01).sum() - 1
#         idx = max(int(idx.item() if torch.is_tensor(idx) else idx), 0)
#         tpr_at1 = tpr[idx].item()
#         asr = ((tpr + 1 - fpr) / 2).max().item()

#         auroc_k.append(auroc)
#         tpr1_k.append(tpr_at1)
#         asr_k.append(asr)
#         auc_mtr.reset(); roc_mtr.reset()

#     print(f"AUROC  per‑step : {auroc_k}")
#     print(f"TPR@1% per‑step : {tpr1_k}")
#     print(f"ASR     per‑step: {asr_k}")
#     print("\nBest over K steps")
#     print(f"  AUROC  = {max(auroc_k):.4f}")
#     print(f"  ASR    = {max(asr_k):.4f}")
#     print(f"  TPR@1% = {max(tpr1_k):.4f}")


# =============================
# Argparse & main
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="LDM MIA single‑file framework: train + probe with resume")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "celeba"], help="Dataset name")
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
    p.add_argument("--epochs_vae", type=int, default=120)
    p.add_argument("--epochs_vqvae", type=int, default=180)
    p.add_argument("--epochs_ldm", type=int, default=2048)

    # Stage toggles
    p.add_argument("--skip_vae", action="store_true")
    p.add_argument("--skip_vqvae", action="store_true")
    p.add_argument("--skip_ldm_vae", action="store_true")
    p.add_argument("--skip_ldm_vq", action="store_true")

    # # Run mode
    # p.add_argument("--mode", type=str, default="train", choices=["train", "probe", "both"],
    #                help="train: only train, probe: only probe from ckpts, both: train then probe")

    # Resume / load checkpoints
    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--vqvae_ckpt", type=str, default=None)
    p.add_argument("--unet_vae_ckpt", type=str, default=None)
    p.add_argument("--unet_vq_ckpt", type=str, default=None)

    # # Probe settings
    # p.add_argument("--probe_min_t", type=int, default=0)
    # p.add_argument("--probe_max_t", type=int, default=300)
    # p.add_argument("--probe_step", type=int, default=10)

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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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
    print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")

    # IO dirs
    dirs = ensure_dirs(args.out_dir)
    preview_batch(train_loader, device, os.path.join(args.out_dir, "sample_batch.png"))

    # Models
    vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base, drop=args.dropout).to(device)
    vqvae = VQVAE(
        in_ch=args.in_channels,
        latent_ch=args.latent_channels,
        base=args.ae_base,
        drop=args.dropout,
        n_embed=args.codebook_size,
        beta=args.vq_beta,
    ).to(device)

    # Resume encoders
    if args.vae_ckpt and os.path.isfile(args.vae_ckpt):
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
        print(f"[Resume] VAE from {args.vae_ckpt}")
    if args.vqvae_ckpt and os.path.isfile(args.vqvae_ckpt):
        vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
        print(f"[Resume] VQ‑VAE from {args.vqvae_ckpt}")

    # TRAIN MODE OR BOTH
    # if args.mode in ("train", "both"):
    if not args.skip_vae:
        print("VAE params:", count_params(vae))
        train_vae_loop(
            vae, train_loader, epochs=args.epochs_vae, lr=args.lr, betas=tuple(args.betas), wd=args.weight_decay,
            out_dir=dirs["vae"], device=device, kl_weight=args.kl_beta
        )
    if not args.skip_vqvae:
        print("VQ‑VAE params:", count_params(vqvae))
        train_vqvae_loop(
            vqvae, train_loader, epochs=args.epochs_vqvae, lr=args.lr, betas=tuple(args.betas), wd=args.weight_decay,
            out_dir=dirs["vqvae"], device=device, vq_lambda=args.vq_lambda
        )

    # LDM (VAE path)
    ldm_vae = LDMTrainer(
        encoder=vae, decoder=vae, use_vq=False, in_latent_ch=args.latent_channels, img_size=args.img_size,
        device=device,
        model_ch=args.unet_model_ch,
        channel_mult=tuple(args.unet_channel_mult),
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.dropout,
        conv_resample=not args.no_conv_resample,
        T=args.T, beta_start=args.beta_start, beta_end=args.beta_end,
        lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay,
        unet_ckpt_path=args.unet_vae_ckpt,
    )
    if not args.skip_ldm_vae:
        print("CompVis LDM (VAE) UNet params:", count_params(ldm_vae.unet))
        ldm_vae.fit(train_loader, epochs=args.epochs_ldm, out_dir=dirs["ldm_vae"])
        ldm_vae.sample_ddpm(n=36, out_dir=dirs["ldm_vae"])  # optional
        ldm_vae.sample_ddim(n=36, steps=args.ddim_steps, out_dir=dirs["ldm_vae"])  # optional

    # LDM (VQ path)
    class VQEncoderInterface(nn.Module):
        def __init__(self, vqvae_model):
            super().__init__()
            self.vq = vqvae_model
        def encode(self, x):
            return self.vq.encode(x)
        def decode(self, z_e):
            z_q, _, _ = self.vq.quant(z_e)   # quantize AFTER diffusion
            return self.vq.decode(z_q)
    vq_if = VQEncoderInterface(vqvae).to(device)

    ldm_vq = LDMTrainer(
        encoder=vq_if, decoder=vq_if, use_vq=True, in_latent_ch=args.latent_channels, img_size=args.img_size,
        device=device,
        model_ch=args.unet_model_ch,
        channel_mult=tuple(args.unet_channel_mult),
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.dropout,
        conv_resample=not args.no_conv_resample,
        T=args.T, beta_start=args.beta_start, beta_end=args.beta_end,
        lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay,
        unet_ckpt_path=args.unet_vq_ckpt,
    )
    if not args.skip_ldm_vq:
        print("CompVis LDM (VQ) UNet params:", count_params(ldm_vq.unet))
        # shorter train by default than VAE path if you wish; here we keep epochs_ldm for symmetry
        ldm_vq.fit(train_loader, epochs=args.epochs_ldm, out_dir=dirs["ldm_vq"])  # optional shorter
        ldm_vq.sample_ddpm(n=36, out_dir=dirs["ldm_vq"])  # optional
        ldm_vq.sample_ddim(n=36, steps=args.ddim_steps, out_dir=dirs["ldm_vq"])  # optional

    # # PROBE MODE OR BOTH
    # if args.mode in ("probe", "both"):
    #     # If paths not provided, try to infer from out_dir
    #     def auto_ckpt(base, rel):
    #         path = os.path.join(base, rel)
    #         return path if os.path.isfile(path) else None

    #     if args.vae_ckpt is None:
    #         args.vae_ckpt = auto_ckpt(args.out_dir, "vae/vae_last.pt")
    #     if args.vqvae_ckpt is None:
    #         args.vqvae_ckpt = auto_ckpt(args.out_dir, "vqvae/vqvae_last.pt")
    #     if args.unet_vae_ckpt is None:
    #         args.unet_vae_ckpt = auto_ckpt(args.out_dir, "ldm_vae/ldm_last.pt")
    #     if args.unet_vq_ckpt is None:
    #         args.unet_vq_ckpt = auto_ckpt(args.out_dir, "ldm_vq/ldm_last.pt")

    #     # Rebuild models for probing and load weights
    #     vae_p = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base, drop=args.dropout).to(device)
    #     vqvae_p = VQVAE(
    #         in_ch=args.in_channels,
    #         latent_ch=args.latent_channels,
    #         base=args.ae_base,
    #         drop=args.dropout,
    #         n_embed=args.codebook_size,
    #         beta=args.vq_beta,
    #     ).to(device)
    #     unet_vae = UNetCompVis(
    #         in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
    #         channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
    #         dropout=args.dropout, conv_resample=not args.no_conv_resample,
    #     ).to(device)
    #     unet_vq = UNetCompVis(
    #         in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
    #         channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
    #         dropout=args.dropout, conv_resample=not args.no_conv_resample,
    #     ).to(device)

    #     if args.vae_ckpt:
    #         vae_p.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    #         print(f"Loaded VAE from {args.vae_ckpt}")
    #     if args.vqvae_ckpt:
    #         vqvae_p.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
    #         print(f"Loaded VQ‑VAE from {args.vqvae_ckpt}")
    #     if args.unet_vae_ckpt:
    #         unet_vae.load_state_dict(torch.load(args.unet_vae_ckpt, map_location=device))
    #         print(f"Loaded UNet(VAE path) from {args.unet_vae_ckpt}")
    #     if args.unet_vq_ckpt:
    #         unet_vq.load_state_dict(torch.load(args.unet_vq_ckpt, map_location=device))
    #         print(f"Loaded UNet(VQ path) from {args.unet_vq_ckpt}")

    #     # Run probes if ckpts are present
    #     if args.unet_vae_ckpt and args.vae_ckpt:
    #         print("\n[Probe] LDM‑VAE path")
    #         probe_unet(
    #             unet_vae, lambda x: get_latents_vae(vae_p, x),
    #             train_loader, val_loader, device,
    #             probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
    #         )
    #     else:
    #         print("[Probe] Skipping VAE path (missing ckpts)")

    #     if args.unet_vq_ckpt and args.vqvae_ckpt:
    #         print("\n[Probe] LDM‑VQ path")
    #         probe_unet(
    #             unet_vq, lambda x: get_latents_vq(vqvae_p, x),
    #             train_loader, val_loader, device,
    #             probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
    #         )
    #     else:
    #         print("[Probe] Skipping VQ path (missing ckpts)")


if __name__ == "__main__":
    main()

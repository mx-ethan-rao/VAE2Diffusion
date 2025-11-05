
"""
ldm_fid_from_attack.py
----------------------
One-file script that reuses the model & dataloading definitions (VAE/UNet/Diffusion + loaders)
from *this file* (integrated below, copied from your attack.py) to:
  1) Load CIFAR-10 or MNIST *training member split* exactly like your pipeline.
  2) Load VAE + UNet (VAE path) checkpoints.
  3) Generate N samples (DDPM or DDIM) and decode to images in [0,1].
  4) Compute FID between generated samples and the *member* training images.

Usage
-----
python ldm_fid_from_attack.py \\
  --dataset cifar10 --dataset_root pytorch-diffusion/datasets \\
  --split_file /path/to/split.npz \\
  --img_size 32 --in_channels 3 --latent_channels 4 \\
  --vae_ckpt /path/to/outputs/vae/vae_last.pt \\
  --unet_vae_ckpt /path/to/outputs/ldm_vae/unet_last.pt \\
  --sampler ddpm --n_gen 1000 --batch_gen 64

Notes
-----
- Real images for FID are loaded with *evaluation* transforms: Resize + (Grayscale for MNIST) + ToTensor (no normalize).
  This yields tensors in [0,1], as required by FID.
- Generated images are decoded by VAE, clamped to [-1,1], then mapped to [0,1].
- If torchmetrics is unavailable, the script can optionally fall back to clean-fid (if installed).
"""
import os, math, argparse, random
from typing import Tuple, Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

# ---------- FID backends ----------
_HAS_TM = True
try:
    from torchmetrics.image.fid import FrechetInceptionDistance as TMFID
except Exception:
    _HAS_TM = False
_HAS_CLEANFID = False
try:
    import cleanfid
    _HAS_CLEANFID = True
except Exception:
    pass

# =============================
# Utilities (from attack.py)
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

def make_gn(ch: int, max_groups: int = 32):
    g = math.gcd(ch, max_groups)
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, ch)

# =============================
# VAE (from attack.py)
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

# =============================
# UNet + Diffusion (from attack.py)
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
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
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
# Dataloaders (from attack.py) + FID-eval variant
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

# FID real-data variant: no normalization, only ToTensor (=> [0,1])
def build_real_fid_loader(dataset: str, root: str, split_file: str, img_size: int, in_channels: int,
                          batch_size: int, num_workers: int):
    ops: List[transforms.Transform] = [transforms.Resize((img_size, img_size))]
    if dataset.lower() == "mnist":
        ops.append(transforms.Grayscale(num_output_channels=in_channels))
    ops.append(transforms.ToTensor())  # [0,1]
    tf_fid = transforms.Compose(ops)

    if dataset.lower() == "cifar10":
        full = CIFAR10(root=root, train=True, download=True, transform=tf_fid)
    elif dataset.lower() == "mnist":
        full = MNIST(root=root, train=True, download=True, transform=tf_fid)
    else:
        raise ValueError("dataset must be cifar10 or mnist")

    split = np.load(split_file)
    mia_train_idxs = split["mia_train_idxs"].tolist()
    real_dataset = Subset(full, mia_train_idxs)
    return DataLoader(real_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

# =============================
# Main
# =============================
def parse_args():
    p = argparse.ArgumentParser("FID using attack.py loaders & models (integrated)")
    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","mnist"])
    p.add_argument("--dataset_root", type=str, default="pytorch-diffusion/datasets")
    p.add_argument("--split_file", type=str, required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--in_channels", type=int, default=3)
    # Model shape
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--ae_base", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unet_model_ch", type=int, default=128)
    p.add_argument("--unet_channel_mult", type=int, nargs="+", default=[1,2,2,2])
    p.add_argument("--unet_num_res_blocks", type=int, default=2)
    p.add_argument("--no_conv_resample", action="store_true")
    # Diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--sampler", type=str, choices=["ddpm","ddim"], default="ddpm")
    p.add_argument("--ddim_steps", type=int, default=50)
    # Checkpoints
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--unet_vae_ckpt", type=str, required=True)
    # Generation/Eval
    p.add_argument("--n_gen", type=int, default=1000)
    p.add_argument("--batch_gen", type=int, default=64)
    p.add_argument("--batch_fid", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

@torch.no_grad()
def compute_fid_torchmetrics(real_loader, fake_loader, device) -> float:
    fid = TMFID(feature=2048).to(device)

    def to_uint8(img: torch.Tensor) -> torch.Tensor:
        # Convert float [0,1] to uint8 [0,255] if needed
        if img.dtype.is_floating_point:
            img = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        return img

    for x in real_loader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = to_uint8(x)
        fid.update(x.to(device), real=True)

    for x in fake_loader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = to_uint8(x)
        fid.update(x.to(device), real=False)

    return float(fid.compute().cpu().item())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Build real loaders using *the same split* ---
    # train_loader_norm is not used for FID, but demonstrates parity with your pipeline;
    # real_fid_loader is used for FID ([0,1] tensors).
    train_loader_norm, _ = build_dataloaders(
        dataset=args.dataset, root=args.dataset_root, split_file=args.split_file,
        img_size=args.img_size, in_channels=args.in_channels, batch_size=args.batch_fid,
        num_workers=args.num_workers
    )
    real_fid_loader = build_real_fid_loader(
        dataset=args.dataset, root=args.dataset_root, split_file=args.split_file,
        img_size=args.img_size, in_channels=args.in_channels, batch_size=args.batch_fid,
        num_workers=args.num_workers
    )
    print(f"[Data] Member(train) examples for FID: {len(real_fid_loader.dataset)}")

    # --- Models ---
    H = W = args.img_size // 4
    vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base, drop=args.dropout).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()

    unet = UNetCompVis(
        in_ch=args.latent_channels, model_ch=args.unet_model_ch, out_ch=args.latent_channels,
        channel_mult=tuple(args.unet_channel_mult), num_res_blocks=args.unet_num_res_blocks,
        dropout=args.dropout, conv_resample=not args.no_conv_resample
    ).to(device)
    unet.load_state_dict(torch.load(args.unet_vae_ckpt, map_location=device))
    unet.eval()

    sched = DiffusionSchedule(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    # --- Generate fake images ---
    n = args.n_gen
    batch = args.batch_gen
    fakes = []
    total = 0
    with torch.no_grad():
        while total < n:
            bs = min(batch, n - total)
            shape = (bs, args.latent_channels, H, W)
            if args.sampler == "ddim":
                lat = sched.sample_ddim(unet, shape, steps=args.ddim_steps, device=device)
            else:
                lat = sched.sample_ddpm(unet, shape, device=device)
            imgs = vae.decode(lat).clamp(-1, 1)
            imgs = (imgs + 1) * 0.5  # -> [0,1]
            fakes.append(imgs.cpu())
            total += bs
            print(f"[Gen] {total}/{n}")

    fake_tensor = torch.cat(fakes, dim=0)
    fake_loader = DataLoader(fake_tensor, batch_size=args.batch_fid, shuffle=False)

    # --- FID ---
    if not _HAS_TM and not _HAS_CLEANFID:
        raise RuntimeError("Install torchmetrics or clean-fid to compute FID.")
    if _HAS_TM:
        fid = compute_fid_torchmetrics(real_fid_loader, fake_loader, device)
        print(f"FID (torchmetrics) = {fid:.4f}")
    else:
        # Export minimal temp dirs for clean-fid
        import tempfile, shutil
        from torchvision.utils import save_image
        tmp_real = tempfile.mkdtemp(prefix="real_")
        tmp_fake = tempfile.mkdtemp(prefix="fake_")
        try:
            rcnt = 0
            for b in real_fid_loader:
                x = b[0] if isinstance(b,(list,tuple)) else b
                for i in range(x.size(0)):
                    save_image(x[i], os.path.join(tmp_real, f"r_{rcnt:06d}.png"))
                    rcnt += 1
            for i in range(fake_tensor.size(0)):
                save_image(fake_tensor[i], os.path.join(tmp_fake, f"f_{i:06d}.png"))
            fid = float(cleanfid.compute_fid(tmp_real, tmp_fake))
            print(f"FID (clean-fid) = {fid:.4f}")
        finally:
            shutil.rmtree(tmp_real, ignore_errors=True)
            shutil.rmtree(tmp_fake, ignore_errors=True)

if __name__ == "__main__":
    main()

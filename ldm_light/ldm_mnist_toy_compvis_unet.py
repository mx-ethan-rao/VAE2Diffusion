
# @title Imports & Config (no val set)
import os, math, random
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torchmetrics.classification import BinaryAUROC, BinaryROC
from torchvision.datasets import MNIST, CIFAR10



import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

@dataclass
class Config:
    # train_root: str = "pytorch-diffusion/datasets/mnist_half/train"
    # val_root: str = "pytorch-diffusion/datasets/mnist_half/val"
    img_size: int = 32
    in_channels: int = 3
    batch_size: int = 128
    num_workers: int = 2

    weight_decay: float = 1e-4
    dropout: float = 0.1
    epochs_vae: int = 250
    epochs_vqvae: int = 350
    epochs_ldm: int = 4096

    latent_channels: int = 4
    codebook_size: int = 512

    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    ddim_steps: int = 50

    out_dir: str = "outputs_ldm_toy_compvis"

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

transform_train = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1) if cfg.in_channels == 1 else transforms.Lambda(lambda x: x),
    transforms.Resize((cfg.img_size, cfg.img_size)),
    # transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*1 if cfg.in_channels==1 else [0.5]*3,
                         [0.5]*1 if cfg.in_channels==1 else [0.5]*3),
])

# train_ds = ImageFolder(cfg.train_root, transform=transform_train)
# train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
#                           num_workers=cfg.num_workers, pin_memory=True)

# val_ds = ImageFolder(cfg.val_root, transform=transform_train)
# val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
#                           num_workers=cfg.num_workers, pin_memory=True)

# Load full MNIST train split
full_train = CIFAR10(root="pytorch-diffusion/datasets", train=True, download=True, transform=transform_train)

# Load your MIA split indices
split = np.load("/banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz")
mia_train_idxs = split["mia_train_idxs"]
mia_val_idxs = split["mia_eval_idxs"]

# print(mia_train_idxs.shape)
# Subset the full MNIST train set into train/val per your split
train_dataset = Subset(full_train, mia_train_idxs.tolist())
val_dataset   = Subset(full_train, mia_val_idxs.tolist())

# print(len(train_dataset))
# DataLoaders (keep your existing params)
train_loader = DataLoader(
    train_dataset, batch_size=cfg.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True
)
# print(len(train_loader))
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")


def save_grid(tensor, path, nrow=8, normalize=True, value_range=(-1,1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, path)

@torch.no_grad()
def show_batch(data_loader, max_images=32, path=None):
    x, y = next(iter(data_loader))
    x = x[:max_images]
    grid = make_grid(x, nrow=8, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(6,6))
    plt.imshow(np.transpose(grid.cpu().numpy(), (1,2,0)))
    plt.axis("off")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
    plt.show()

show_batch(train_loader, path=os.path.join(cfg.out_dir, "sample_batch.png"))


# @title Utils (param count + GN helper)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_gn(ch: int, max_groups: int = 32):
    g = math.gcd(ch, max_groups)
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, ch)


# @title VAE
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
    def __init__(self, in_ch=1, latent_ch=4, base=64, drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base, 3, stride=2, padding=1),  # 32->16
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base*2, 3, stride=2, padding=1),# 16->8
            ResBlockSimple(base*2, drop),
        )
        self.to_mu = nn.Conv2d(base*2, latent_ch, 1)
        self.to_logvar = nn.Conv2d(base*2, latent_ch, 1)

        self.dec_in = nn.Conv2d(latent_ch, base*2, 1)
        self.dec = nn.Sequential(
            ResBlockSimple(base*2, drop),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1), # 8->16
            ResBlockSimple(base, drop),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),   # 16->32
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

vae = VAE(in_ch=cfg.in_channels, latent_ch=cfg.latent_channels, base=128, drop=cfg.dropout).to(device)
print("VAE params:", count_params(vae))


# @title Train VAE (train-only)
def train_vae(model, train_loader, epochs=6, lr=2e-4, wd=1e-4, out_dir="outputs_ldm_toy_compvis/vae"):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=cfg.betas, weight_decay=wd)

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x, deterministic=False)
            loss = recon_loss(x, recon) + 1e-3 * kld_loss(mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)
        print(f"[VAE] Epoch {epoch}: train {tr_loss:.4f}")

        with torch.no_grad():
            x,_ = next(iter(train_loader))
            x = x[:32].to(device)
            recon, _, _, _ = model(x, deterministic=True)
            save_grid(torch.cat([x, recon], dim=0).cpu(), os.path.join(out_dir, f"recon_e{epoch}.png"), nrow=8)

    torch.save(model.state_dict(), os.path.join(out_dir, "vae_last.pt"))
    return model

vae = train_vae(vae, train_loader, epochs=cfg.epochs_vae, lr=cfg.lr, wd=cfg.weight_decay)


# @title VQ-VAE
class VectorQuantizer(nn.Module):
    def __init__(self, n_embed=256, embed_dim=4, beta=0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z = z_e.permute(0,2,3,1).contiguous()
        z_flat = z.view(-1, C)
        emb = self.embedding.weight
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ emb.t()
                + emb.pow(2).sum(1, keepdim=True).t())
        _, idx = torch.min(dist, dim=1)
        z_q = emb[idx].view(B, H, W, C).permute(0,3,1,2).contiguous()

        z_q_st = z_e + (z_q - z_e).detach()
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commit_loss   = F.mse_loss(z_q, z_e.detach())
        loss = codebook_loss + self.beta * commit_loss
        return z_q_st, loss, idx.view(B, H, W)

class VQVAE(nn.Module):
    def __init__(self, in_ch=1, latent_ch=4, base=64, drop=0.1, n_embed=256, beta=0.25):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base, 3, stride=2, padding=1),  # 32->16
            ResBlockSimple(base, drop),
            nn.Conv2d(base, latent_ch, 3, stride=2, padding=1), # 16->8
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

    def encode(self, x):
        return self.enc(x)
    def decode(self, z_q):
        return self.dec(z_q)
    def forward(self, x):
        z_e = self.encode(x)
        z_q, vq_loss, _ = self.quant(z_e)
        recon = self.decode(z_q)
        return recon, vq_loss, z_q

vqvae = VQVAE(in_ch=cfg.in_channels, latent_ch=cfg.latent_channels, base=128,
              drop=cfg.dropout, n_embed=cfg.codebook_size, beta=0.25).to(device)
print("VQ-VAE params:", count_params(vqvae))


# @title Train VQ-VAE (train-only)
def train_vqvae(model, train_loader, epochs=8, lr=2e-4, wd=1e-4, out_dir="outputs_ldm_toy_compvis/vqvae"):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=cfg.betas, weight_decay=wd)

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, vq_loss, _ = model(x)
            loss = F.l1_loss(recon, x) + 0.5 * vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)
        print(f"[VQ-VAE] Epoch {epoch}: train {tr_loss:.4f}")

        with torch.no_grad():
            x,_ = next(iter(train_loader))
            x = x[:32].to(device)
            recon, _, _ = model(x)
            save_grid(torch.cat([x, recon], dim=0).cpu(), os.path.join(out_dir, f"recon_e{epoch}.png"), nrow=8)

    torch.save(model.state_dict(), os.path.join(out_dir, "vqvae_last.pt"))
    return model

vqvae = train_vqvae(vqvae, train_loader, epochs=cfg.epochs_vqvae, lr=cfg.lr, wd=cfg.weight_decay)


# @title CompVis-like UNet with concat skips
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

def normalization(ch):
    g = math.gcd(ch, 32)
    if g <= 0: g = 1
    return nn.GroupNorm(g, ch)

class TimestepBlock(nn.Module):
    def forward(self, x, emb): raise NotImplementedError

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
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_ch, out_ch),
        )
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
    def __init__(self,
                 in_ch=4,
                 model_ch=64,
                 out_ch=4,
                 channel_mult=(1, 2),
                 num_res_blocks=1,
                 dropout=0.1,
                 conv_resample=True):
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
                layers = [ResBlock(ch, self.time_dim, dropout, out_ch=mult*model_ch)]
                ch = mult * model_ch
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, use_conv=conv_resample)))
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
            normalization(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1)
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



# @title Diffusion schedule & samplers (DDPM / DDIM)
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
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1,1,1,1)
        sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1,1,1,1)
        sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1,1,1,1)
        return (x_t - sqrt_1mab * eps) / (sqrt_ab + 1e-8)

    def p_sample_ddpm(self, model, x_t, t):
        betas_t = self.betas[t].view(-1,1,1,1)
        eps = model(x_t, t)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            mean = (1.0/torch.sqrt(1.0 - betas_t)) * (x_t - betas_t/torch.sqrt(1.0 - self.alpha_bars[t]).view(-1,1,1,1) * eps)
            sample = mean + torch.sqrt(betas_t) * noise
        else:
            sample = self.predict_x0_from_eps(x_t, t, eps)
        return sample

    @torch.no_grad()
    def sample_ddpm(self, model, shape, device="cpu"):
        B, C, H, W = shape
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            x = self.p_sample_ddpm(model, x, t)
        return x

    @torch.no_grad()
    def sample_ddim(self, model, shape, steps=50, device="cpu"):
        B, C, H, W = shape
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T-1, 0, steps, device=device).long()
        for i in range(steps):
            t = ts[i].repeat(B)
            eps = model(x, t)
            ab_t = self.alpha_bars[t].view(-1,1,1,1)
            x0 = self.predict_x0_from_eps(x, t, eps)
            if i == steps - 1:
                x = x0
                break
            t_next = ts[i+1].repeat(B)
            ab_t_next = self.alpha_bars[t_next].view(-1,1,1,1)
            x = torch.sqrt(ab_t_next) * x0 + torch.sqrt(1 - ab_t_next) * eps
        return x


# @title LDM Trainer using CompVis UNet (train-only)
class LDMTrainer:
    def __init__(self, encoder, decoder, use_vq=False):
        self.encoder = encoder.eval().to(device)
        self.decoder = decoder.eval().to(device)
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.decoder.parameters(): p.requires_grad = False

        self.unet = UNetCompVis(in_ch=cfg.latent_channels, model_ch=128, out_ch=cfg.latent_channels,
                                channel_mult=(1, 2, 2, 2), num_res_blocks=2, dropout=cfg.dropout).to(device)
        self.opt = torch.optim.AdamW(self.unet.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        self.sched = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=device)
        self.use_vq = use_vq

    @torch.no_grad()
    def get_latents(self, x):
        if self.use_vq:
            z_e = self.encoder.encode(x)
            z_q, _, _ = self.decoder.quant(z_e) if hasattr(self.decoder, "quant") else self.encoder.quant(z_e)
            return z_q
        else:
            mu, logvar = self.encoder.encode(x)
            return mu

    def train_epoch(self, loader):
        self.unet.train()
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                z = self.get_latents(x)

            B = z.size(0)
            t = torch.randint(0, self.sched.T, (B,), device=device).long()
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
    
    def probe(self, train_loader, val_loader, t):
        self.unet.eval()
        pred_m_batch = []
        pred_n_batch = []
        for (m, _), (n, _) in zip(train_loader, val_loader):
            m = m.to(device)
            n = n.to(device)
            with torch.no_grad():
                z_m = self.get_latents(m)
                z_n = self.get_latents(n)

            B = z_m.size(0)

            t_batch = torch.ones([B,], device=device).long() * t
            with torch.no_grad():
                pred_m = self.unet(z_m, t_batch)
                pred_n = self.unet(z_n, t_batch)
            pred_m_batch.append((pred_m.abs()**4).flatten(1).sum(dim=-1))
            pred_n_batch.append((pred_n.abs()**4).flatten(1).sum(dim=-1))
            pred_m_batch = [torch.cat(pred_m_batch)]
            pred_n_batch = [torch.cat(pred_n_batch)]
            
        return pred_m_batch[0], pred_n_batch[0]

    def fit(self, train_loader, epochs=10, out_dir="outputs_ldm_toy_compvis/ldm_vae"):
        os.makedirs(out_dir, exist_ok=True)
        for epoch in range(1, epochs+1):
            tr = self.train_epoch(train_loader)
            print(f"[LDM] Epoch {epoch}: train {tr:.4f}")
            torch.save(self.unet.state_dict(), os.path.join(out_dir, "ldm_last.pt"))

    @torch.no_grad()
    def sample_ddpm(self, n=36, out_dir="outputs_ldm_toy_compvis/ldm_vae"):
        self.unet.eval()
        latents = self.sched.sample_ddpm(self.unet, (n, cfg.latent_channels, cfg.img_size//4, cfg.img_size//4), device=device)
        imgs = self.decoder.decode(latents).clamp(-1, 1).cpu()
        save_grid(imgs, os.path.join(out_dir, "samples_ddpm.png"), nrow=6)
        return imgs

    @torch.no_grad()
    def sample_ddim(self, n=36, steps=50, out_dir="outputs_ldm_toy_compvis/ldm_vae"):
        self.unet.eval()
        latents = self.sched.sample_ddim(self.unet, (n, cfg.latent_channels, cfg.img_size//4, cfg.img_size//4),
                                         steps=steps, device=device)
        imgs = self.decoder.decode(latents).clamp(-1, 1).cpu()
        save_grid(imgs, os.path.join(out_dir, "samples_ddim.png"), nrow=6)
        return imgs


# @title Train LDMs (VAE & VQ-VAE latents) and sample
vae_encoder = vae; vae_decoder = vae
ldm_vae = LDMTrainer(encoder=vae_encoder, decoder=vae_decoder, use_vq=False)
print("CompVis LDM(vae) params:", count_params(ldm_vae.unet))
ldm_vae.fit(train_loader, epochs=cfg.epochs_ldm, out_dir=os.path.join(cfg.out_dir, "ldm_vae"))
_ = ldm_vae.sample_ddpm(n=36, out_dir=os.path.join(cfg.out_dir, "ldm_vae"))
_ = ldm_vae.sample_ddim(n=36, steps=cfg.ddim_steps, out_dir=os.path.join(cfg.out_dir, "ldm_vae"))

class VQEncoderWrapper(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.vq = vqvae
    def encode(self, x):
        return self.vq.encode(x)
    def quant(self, z_e):
        return self.vq.quant(z_e)

vq_enc_wrap = VQEncoderWrapper(vqvae).to(device)
ldm_vq = LDMTrainer(encoder=vq_enc_wrap, decoder=vqvae, use_vq=True)
print("CompVis LDM(vq) params:", count_params(ldm_vq.unet))
ldm_vq.fit(train_loader, epochs=max(6, cfg.epochs_ldm//2), out_dir=os.path.join(cfg.out_dir, "ldm_vq"))
_ = ldm_vq.sample_ddpm(n=36, out_dir=os.path.join(cfg.out_dir, "ldm_vq"))
_ = ldm_vq.sample_ddim(n=36, steps=cfg.ddim_steps, out_dir=os.path.join(cfg.out_dir, "ldm_vq"))


# @title Sanity check: UNet IO shapes
with torch.no_grad():
    unet = UNetCompVis(in_ch=cfg.latent_channels, model_ch=64, out_ch=cfg.latent_channels,
                       channel_mult=(1,2), num_res_blocks=1, dropout=cfg.dropout).to(device)
    x = torch.randn(2, cfg.latent_channels, cfg.img_size//4, cfg.img_size//4, device=device)  # [2,4,8,8]
    t = torch.randint(0, cfg.T, (2,), device=device)
    y = unet(x, t)
    print("UNet in/out:", x.shape, "->", y.shape)


        

probe_t = range(0, 300, 10)
pred_ms = []
pred_ns = []
for p_t in probe_t:
    m, n = ldm_vae.probe(train_loader, val_loader, p_t)
    # print(m.shape)
    pred_ms.append(m)
    pred_ns.append(n)
# print(pred_ms)
# print(pred_ns)
pred_ms = torch.stack(pred_ms)
pred_ns = torch.stack(pred_ns)
print(pred_ms.shape)
print(pred_ns.shape)

# ⑤ metrics
auc_mtr, roc_mtr = BinaryAUROC().to(device), BinaryROC().to(device)
auroc_k, tpr1_k, asr_k = [], [], []

for k in range(pred_ms.size(0)):
    m, n = pred_ms[k], pred_ns[k]
    scale = torch.max(m.max(), n.max()); m, n = m/scale, n/scale
    scores = torch.cat([m, n])
    labels = torch.cat([torch.zeros_like(m), torch.ones_like(n)]).long()

    auroc = auc_mtr(scores, labels).item()
    fpr, tpr, _ = roc_mtr(scores, labels)
    idx = (fpr < 0.01).sum() - 1
    tpr_at1 = tpr[idx].item()
    asr = ((tpr + 1 - fpr) / 2).max().item()

    auroc_k.append(auroc); tpr1_k.append(tpr_at1); asr_k.append(asr)
    auc_mtr.reset(); roc_mtr.reset()

print('AUROC  per-step :', auroc_k)
print('TPR@1% per-step :', tpr1_k)
print('ASR     per-step:', asr_k)
print('\nBest over K steps')
print(f'  AUROC  = {max(auroc_k):.4f}')
print(f'  ASR    = {max(asr_k):.4f}')
print(f'  TPR@1% = {max(tpr1_k):.4f}')


probe_t = range(0, 300, 10)
pred_ms = []
pred_ns = []
for p_t in probe_t:
    m, n = ldm_vq.probe(train_loader, val_loader, p_t)
    # print(m.shape)
    pred_ms.append(m)
    pred_ns.append(n)
# print(pred_ms)
# print(pred_ns)
pred_ms = torch.stack(pred_ms)
pred_ns = torch.stack(pred_ns)
print(pred_ms.shape)
print(pred_ns.shape)

# ⑤ metrics
auc_mtr, roc_mtr = BinaryAUROC().to(device), BinaryROC().to(device)
auroc_k, tpr1_k, asr_k = [], [], []

for k in range(pred_ms.size(0)):
    m, n = pred_ms[k], pred_ns[k]
    scale = torch.max(m.max(), n.max()); m, n = m/scale, n/scale
    scores = torch.cat([m, n])
    labels = torch.cat([torch.zeros_like(m), torch.ones_like(n)]).long()

    auroc = auc_mtr(scores, labels).item()
    fpr, tpr, _ = roc_mtr(scores, labels)
    idx = (fpr < 0.01).sum() - 1
    tpr_at1 = tpr[idx].item()
    asr = ((tpr + 1 - fpr) / 2).max().item()

    auroc_k.append(auroc); tpr1_k.append(tpr_at1); asr_k.append(asr)
    auc_mtr.reset(); roc_mtr.reset()

print('AUROC  per-step :', auroc_k)
print('TPR@1% per-step :', tpr1_k)
print('ASR     per-step:', asr_k)
print('\nBest over K steps')
print(f'  AUROC  = {max(auroc_k):.4f}')
print(f'  ASR    = {max(asr_k):.4f}')
print(f'  TPR@1% = {max(tpr1_k):.4f}')
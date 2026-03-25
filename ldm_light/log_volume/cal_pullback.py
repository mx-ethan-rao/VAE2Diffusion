#!/usr/bin/env python3
"""
compute_logvol.py
Compute per-sample Jacobian log-volumes for VAE decoder on the MIA train/val splits
and save results to a local .npz file.

Run:
  python compute_logvol.py --split-file path/to/splits.npz --vae-ckpt path/to/vae_last.pt --out npz_out.npz
"""
import os
import math
import argparse
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# ----------------------------
# Minimal model definitions (VAE + small UNet if needed)
# Only VAE required for compute script.
# ----------------------------

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
    
def make_gn(ch: int, max_groups: int = 32):
    g = math.gcd(ch, max_groups)
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, ch)

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

# ----------------------------
# Data helpers
# ----------------------------
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
        transforms.ToTensor(),
        transforms.Normalize([0.5] * in_channels, [0.5] * in_channels),
    ]
    return transforms.Compose(ops)

def build_dataloaders(dataset: str, root: str, split_file: str, img_size: int, in_channels: int,
                      batch_size: int, num_workers: int, shuffle_train=False):
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
    mia_train_idxs = split["mia_train_idxs"]
    mia_val_idxs = split["mia_eval_idxs"]

    train_dataset = Subset(full_train, mia_train_idxs)
    val_dataset = Subset(full_train, mia_val_idxs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, mia_train_idxs, mia_val_idxs

# ----------------------------
# Jacobian / log-volume helpers
# ----------------------------
def topk_singular_values_power(J: torch.Tensor, k=8, n_iter=10):
    """
    Estimate top-k singular values of J (shape M x d) using randomized power iterations on J^T J.
    Returns singular values in descending order (k,).
    """
    device = J.device
    d = J.shape[1]
    # random initial vectors (d x k)
    Q = torch.randn(d, k, device=device)
    for _ in range(n_iter):
        Z = J @ Q        # (M x k)
        Q = J.T @ Z      # (d x k)  -> multiply by J^T J implicitly
        Q, _ = torch.linalg.qr(Q)  # orthonormalize
    # Rayleigh matrix
    Z = J @ Q          # (M x k)
    B = Q.T @ (J.T @ Z)  # (k x k)
    eigvals = torch.linalg.eigvalsh(B)
    eigvals = torch.sort(eigvals.real)[0].flip(0)
    eigvals = eigvals.clamp(min=1e-12)
    svals = eigvals.sqrt()
    return svals[:k]

def compute_log_volume_per_sample_decoder(
    decoder: nn.Module,
    z_tensor: torch.Tensor,
    device: torch.device,
    max_samples: Optional[int] = None,
    k: int = 8,
    n_iter: int = 10,
    desc: str = "compute log-vol"
):
    """
    For each z in z_tensor (shape N x C x H x W or N x C), compute approximate log-volume
    as sum(log(top-k singular values of Jacobian(decoder.decode) at z)).
    Returns a numpy array shape (N_selected,).
    """
    decoder = decoder.to(device).eval()
    N = z_tensor.shape[0]
    if max_samples is not None:
        N = min(N, max_samples)
        z_tensor = z_tensor[:N]

    results = []
    for i in tqdm(range(N), desc=desc):
        z = z_tensor[i:i+1].to(device).detach().clone()
        z.requires_grad_(True)

        def decode_flat(z_in):
            out = decoder.decode(z_in)      # (1, Cx, Hx, Wx)
            return out.view(-1)             # flatten to (M,)

        # jacobian: shape (M, 1, in_dim) for single input -> reshape to (M, in_dim)
        J = jacobian(decode_flat, z)
        # jacobian returns shape (M, *z.shape) where z.shape=(1,C,H,W) etc -> flatten latent dims to in_dim
        J = J.reshape(J.shape[0], -1).to(device)

        # approximate top-k singular values
        svals = topk_singular_values_power(J, k=k, n_iter=n_iter)
        log_vol = torch.log(svals).sum().item()
        results.append(float(log_vol))

        # cleanup
        z.requires_grad_(False)
        del J, svals
        torch.cuda.empty_cache()

    return np.array(results, dtype=np.float32)

# ----------------------------
# helpers to encode latents (we store latents on CPU to allow offline checking if needed)
# ----------------------------
def build_latent_arrays_for_dataset(encode_fn, loader, device, max_samples=None):
    latent_list = []
    count = 0
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            mu, _ = encode_fn(x)  # encode returns (mu, logvar) in our VAE encode wrapper
            # keep mu on CPU
            latent_list.append(mu.cpu())
        count += mu.size(0)
        if max_samples is not None and count >= max_samples:
            break
    Z = torch.cat(latent_list, dim=0)
    if max_samples is not None and Z.size(0) > max_samples:
        Z = Z[:max_samples]
    return Z

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "celeba"])
    p.add_argument("--dataset-root", type=str, default="pytorch-diffusion/datasets")
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="logvols_splits.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--ae-base", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None, help="Limit number samples per split (for testing)")
    p.add_argument("--k", type=int, default=8, help="Top-k singular values to use")
    p.add_argument("--n-iter", type=int, default=10, help="Power iteration rounds")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # build loaders with deterministic order (shuffle=False) so saved arrays align to split indices
    train_loader, val_loader, train_idxs, val_idxs = build_dataloaders(
        dataset=args.dataset,
        root=args.dataset_root,
        split_file=args.split_file,
        img_size=args.img_size,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False
    )

    print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")

    # load vae
    vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base).to(device)
    state = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(state)
    vae.eval()
    print("Loaded VAE:", args.vae_ckpt)

    # encode latents (mu) for each dataset (kept on CPU)
    print("Encoding train latents...")
    def encode_fn(x):
        return vae.encode(x)  # returns (mu, logvar)
    Z_train = build_latent_arrays_for_dataset(encode_fn, train_loader, device, max_samples=args.max_samples)
    print("Encoding val latents...")
    Z_val = build_latent_arrays_for_dataset(encode_fn, val_loader, device, max_samples=args.max_samples)

    print("Train latents shape:", Z_train.shape)
    print("Val latents shape:", Z_val.shape)

    # compute log-volumes (this is the heavy step)
    print("Computing train log-volumes (may be slow)...")
    logv_train = compute_log_volume_per_sample_decoder(vae, Z_train, device, max_samples=None, k=args.k, n_iter=args.n_iter, desc="train logvol")
    print("Computing val log-volumes (may be slow)...")
    logv_val = compute_log_volume_per_sample_decoder(vae, Z_val, device, max_samples=None, k=args.k, n_iter=args.n_iter, desc="val logvol")

    # Save arrays along with original split indices so probe step can map easily
    np.savez_compressed(
        args.out,
        logv_train=logv_train.astype(np.float32),
        logv_val=logv_val.astype(np.float32),
        train_indices=np.asarray(train_idxs),
        val_indices=np.asarray(val_idxs),
    )
    print(f"Saved log-volumes to {args.out}")
    print("Done.")

if __name__ == "__main__":
    main()

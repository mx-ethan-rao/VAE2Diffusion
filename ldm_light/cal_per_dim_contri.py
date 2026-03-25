#!/usr/bin/env python3
"""
compute_logvol.py
Compute per-sample per-dimension pullback metric contributions for a VAE decoder
on the MIA train/val splits and save results to a local .npz file.

Each point z gets per-dimension contributions:
    c_k = 0.5 * log( (J_D(z)^T J_D(z))_{kk} )
computed efficiently using a Hutchinson estimator.

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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

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

# ----------------------------
# Minimal model definitions (VAE)
# ----------------------------
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
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, mia_train_idxs, mia_val_idxs

# ----------------------------
# Encode latents
# ----------------------------
@torch.no_grad()
def build_latent_arrays_for_dataset(encode_fn, loader, device, max_samples=None):
    latent_list = []
    count = 0
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            mu, _ = encode_fn(x)
            latent_list.append(mu.cpu())
        count += mu.size(0)
        if max_samples is not None and count >= max_samples:
            break
    Z = torch.cat(latent_list, dim=0)
    if max_samples is not None and Z.size(0) > max_samples:
        Z = Z[:max_samples]
    return Z

# ----------------------------
# Per-dimension contributions via Hutchinson estimator
# ----------------------------
def compute_per_dim_contrib_decoder(
    decoder: VAE,
    z_tensor: torch.Tensor,
    device: torch.device,
    n_mc: int = 8,
    eps: float = 1e-12,
    desc: str = "per-dim contrib"
):
    """
    For each z in z_tensor, compute per-dimension contributions:
        c_k = 0.5 * log( (J_D^T J_D)_{kk} + eps )
    using Hutchinson estimator:
        diag(J^T J) ≈ E_v[(J^T v) ⊙ (J^T v)], v ~ N(0, I).
    Returns numpy array of shape (N, D) where D = flattened latent dimension.
    """
    decoder = decoder.to(device).eval()
    N = z_tensor.shape[0]
    D = int(np.prod(z_tensor.shape[1:]))

    results = []
    for i in tqdm(range(N), desc=desc):
        z = z_tensor[i:i+1].to(device).detach()
        z.requires_grad_(True)

        out = decoder.decode(z)
        out_flat = out.flatten(start_dim=1)
        B, P = out_flat.shape

        diag_est = torch.zeros((B, D), device=device)

        for _ in range(n_mc):
            v = torch.randn_like(out_flat)
            g = torch.autograd.grad(
                outputs=out_flat,
                inputs=z,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0]
            diag_est += g.flatten(start_dim=1).pow(2)

        diag_est /= float(n_mc)
        per_dim_log = 0.5 * torch.log(diag_est + eps)
        results.append(per_dim_log.detach().cpu())

        z.requires_grad_(False)
        del out, out_flat, diag_est, per_dim_log
        torch.cuda.empty_cache()

    per_dim = torch.cat(results, dim=0)
    return per_dim.numpy().astype(np.float32)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "celeba"])
    p.add_argument("--dataset-root", type=str, default="pytorch-diffusion/datasets")
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="perdim_contribs.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--ae-base", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=8, help="MC samples for diag(J^T J) estimation")
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_idxs, val_idxs = build_dataloaders(
        dataset=args.dataset,
        root=args.dataset_root,
        split_file=args.split_file,
        img_size=args.img_size,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False,
    )

    print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")

    # load VAE
    vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base).to(device)
    state = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(state)
    vae.eval()
    print("Loaded VAE:", args.vae_ckpt)

    # Encode latents
    print("Encoding train latents...")
    def encode_fn(x):
        return vae.encode(x)
    Z_train = build_latent_arrays_for_dataset(encode_fn, train_loader, device, max_samples=args.max_samples)
    print("Encoding val latents...")
    Z_val = build_latent_arrays_for_dataset(encode_fn, val_loader, device, max_samples=args.max_samples)

    print("Train latents shape:", Z_train.shape)
    print("Val latents shape:", Z_val.shape)

    # Compute per-dimension contributions
    print(f"Computing per-dimension contributions (n_mc={args.n_mc}) for train...")
    per_dim_train = compute_per_dim_contrib_decoder(vae, Z_train, device, n_mc=args.n_mc, desc="train per-dim")
    print("Computing per-dimension contributions for val...")
    per_dim_val = compute_per_dim_contrib_decoder(vae, Z_val, device, n_mc=args.n_mc, desc="val per-dim")

    # Save only per-dimension results
    np.savez_compressed(
        args.out,
        per_dim_logcontrib_train=per_dim_train.astype(np.float32),
        per_dim_logcontrib_val=per_dim_val.astype(np.float32),
        train_indices=np.asarray(train_idxs),
        val_indices=np.asarray(val_idxs),
    )
    print(f"Saved per-dimension contributions to {args.out}")
    print("Done.")

if __name__ == "__main__":
    main()

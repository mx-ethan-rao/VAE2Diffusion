#!/usr/bin/env python3
"""
variance_by_group.py

Compute total variance (sum of covariance eigenvalues / trace) of VAE-encoded latents
for different groups defined by a log-volume .npz file (median / quartiles / random split).

Run (example):
  python variance_by_group.py \
    --dataset cifar10 --dataset-root /home/ethanrao/MIA_LDM/data \
    --split-file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
    --vae-ckpt /banana/ethan/MIA_LDM_data/KL_sweep/1e_2/vae/vae_last.pt \
    --logvol /home/ethanrao/MIA_LDM/data/logvols_cifar10_beta_1e_2.npz \
    --grouping quartiles --save-tv tv_out.npz
"""

import argparse
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


# ----------------------------
# reuse the VAE/UNet definitions (only the ones necessary)
# ----------------------------
def make_gn(ch: int, max_groups: int = 32):
    g = math.gcd(ch, max_groups) if 'math' in globals() else 1
    return nn.GroupNorm(g, ch)

# To avoid repeating entire code, reimplement minimal VAE + UNet matching shapes used in your probe.
# (Same as compute_logvol script; ensure architecture matches checkpoint)
import math
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
            nn.Conv2d(base, base, 3, stride=2, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            ResBlockSimple(base * 2, drop),
        )
        self.to_mu = nn.Conv2d(base * 2, latent_ch, 1)
        self.to_logvar = nn.Conv2d(base * 2, latent_ch, 1)
        self.dec_in = nn.Conv2d(latent_ch, base * 2, 1)
        self.dec = nn.Sequential(
            ResBlockSimple(base * 2, drop),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            ResBlockSimple(base, drop),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),
            ResBlockSimple(base, drop),
            make_gn(base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )
    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        return mu, logvar
    def decode(self, z):
        h = self.dec_in(z)
        return self.dec(h)


# ----------------------------
# Data helpers
# ----------------------------
def build_transforms(img_size: int, in_channels: int, dataset_name: str):
    ops: List[transforms.Transform] = [transforms.Resize((img_size, img_size))]
    if dataset_name.lower() == "mnist":
        ops.append(transforms.Grayscale(num_output_channels=in_channels))
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'mnist'.")

    split = np.load(split_file)
    mia_train_idxs = split["mia_train_idxs"]
    mia_val_idxs = split["mia_eval_idxs"]

    train_dataset = Subset(full_train, mia_train_idxs)
    val_dataset = Subset(full_train, mia_val_idxs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def build_subset_loader_from_mask(orig_loader, mask, batch_size=128):
    orig_subset = orig_loader.dataset
    if not isinstance(orig_subset, Subset):
        raise RuntimeError("Expected Subset for orig_loader.dataset")
    base_indices = np.array(orig_subset.indices)
    chosen_indices = base_indices[mask]
    new_subset = Subset(orig_subset.dataset, chosen_indices.tolist())
    return DataLoader(new_subset, batch_size=batch_size, shuffle=False,
                      num_workers=orig_loader.num_workers, pin_memory=True)


# ----------------------------
# Total variance (trace of covariance) of VAE-μ latents
# Streaming version via per-feature sums
# ----------------------------
class RunningSumVar:
    """
    Track per-feature sum and sum of squares to compute variance:
      var = E[x^2] - (E[x])^2  (population or unbiased scaling selectable outside)
    """
    def __init__(self, D: int, device: torch.device):
        self.n = 0
        self.sum = torch.zeros(D, device=device)
        self.sum2 = torch.zeros(D, device=device)

    @torch.no_grad()
    def update(self, X: torch.Tensor):
        # X: [B, D]
        if X.numel() == 0:
            return
        self.n += X.shape[0]
        self.sum += X.sum(dim=0)
        self.sum2 += (X * X).sum(dim=0)

    @torch.no_grad()
    def total_variance(self, unbiased: bool = False) -> float:
        if self.n <= 1:
            return 0.0
        N = float(self.n)
        mean = self.sum / N
        ex2 = self.sum2 / N              # E[x^2]
        var_pop = (ex2 - mean * mean).clamp_min(0.0)  # per-feature pop variance
        if unbiased:
            # Unbiased correction: multiply pop var by N/(N-1)
            var = var_pop * (N / (N - 1.0))
        else:
            var = var_pop
        return var.sum().item()  # trace(cov)

@torch.no_grad()
def online_total_variance_latents(vae: VAE, loader: DataLoader, device: torch.device, unbiased: bool=False) -> Tuple[float, int]:
    vae.eval()
    n_seen = 0
    rsv: RunningSumVar = None
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        mu, _ = vae.encode(imgs)            # [B, C, H, W]
        Z = mu.flatten(1)                   # [B, D]
        if rsv is None:
            rsv = RunningSumVar(Z.shape[1], device=device)
        rsv.update(Z)
        n_seen += Z.shape[0]
    tv = rsv.total_variance(unbiased=unbiased) if rsv is not None else 0.0
    return tv, n_seen

@torch.no_grad()
def compute_total_variance_via_eig(vae: VAE, loader: DataLoader, device: torch.device, unbiased: bool=False) -> float:
    """
    Optional, RAM-heavy: exact covariance, sum eigenvalues.
    """
    vae.eval()
    Zs = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        mu, _ = vae.encode(imgs)
        Zs.append(mu.flatten(1).cpu())
    if len(Zs) == 0:
        return 0.0
    Z = torch.cat(Zs, dim=0)  # [N, D] on CPU
    N = Z.shape[0]
    if N <= 1:
        return 0.0
    Z = Z - Z.mean(dim=0, keepdim=True)
    scale = 1.0 / (N - 1 if unbiased else N)
    cov = scale * (Z.t() @ Z)
    cov = 0.5 * (cov + cov.t())           # symmetrize
    eigvals = torch.linalg.eigvalsh(cov)  # should be >=0
    eigvals = eigvals.clamp_min(0)
    return eigvals.sum().item()

def compute_tv_for_group(vae: VAE,
                         tr_loader: DataLoader,
                         va_loader: DataLoader,
                         device: torch.device,
                         unbiased: bool = False,
                         via_eig: bool = False) -> Dict[str, float]:
    """
    Compute total variance for member-only, heldout-only, and both combined.
    """
    tv_member, n_m = online_total_variance_latents(vae, tr_loader, device, unbiased=unbiased)
    tv_heldout, n_h = online_total_variance_latents(vae, va_loader, device, unbiased=unbiased)

    # Combined (run streaming again over chained loader)
    class ChainLoader:
        def __init__(self, loaders): self.loaders = loaders
        def __iter__(self):
            for L in self.loaders:
                for batch in L: yield batch
        def __len__(self): return sum(len(L) for L in self.loaders)
        @property
        def num_workers(self): return getattr(self.loaders[0], "num_workers", 0)
        @property
        def dataset(self): return self.loaders[0].dataset

    tv_both, n_b = online_total_variance_latents(vae, ChainLoader([tr_loader, va_loader]),
                                                 device, unbiased=unbiased)

    out = {
        "TV_member": tv_member,
        "TV_heldout": tv_heldout,
        "TV_both": tv_both,
        "N_member": float(n_m),
        "N_heldout": float(n_h),
        "N_both": float(n_b),
    }

    if via_eig:
        eig_member = compute_total_variance_via_eig(vae, tr_loader, device, unbiased=unbiased)
        eig_heldout = compute_total_variance_via_eig(vae, va_loader, device, unbiased=unbiased)
        eig_both = compute_total_variance_via_eig(vae, ChainLoader([tr_loader, va_loader]),
                                                  device, unbiased=unbiased)
        out.update({
            "TV_member_eig": eig_member,
            "TV_heldout_eig": eig_heldout,
            "TV_both_eig": eig_both,
        })
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    p.add_argument("--dataset-root", type=str, default="/home/ethanrao/MIA_LDM/data")
    p.add_argument("--split-file", type=str, default="/banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz", required=False)
    p.add_argument("--vae-ckpt", type=str, default="/banana/ethan/MIA_LDM_data/KL_sweep/1e_2/vae/vae_last.pt", required=False)
    p.add_argument("--logvol", type=str, default="/home/ethanrao/MIA_LDM/data/logvols_cifar10_beta_1e_2.npz", required=False,
                   help=".npz produced by compute_logvol.py")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--ae-base", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grouping", type=str, default="median",
                   choices=["median", "quartiles", "random_split"], help="Grouping strategy")
    # Variance options
    p.add_argument("--tv-unbiased", action="store_true",
                   help="Use unbiased variance (divide by N-1). Default: population variance (divide by N).")
    p.add_argument("--tv-via-eig", action="store_true",
                   help="Also compute total variance via eigenvalue sum (slow/memory-heavy).")
    p.add_argument("--save-tv", type=str, default="",
                   help="Path to save per-group total-variance results (.npz).")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = build_dataloaders(
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

    # log-volume file
    data = np.load(args.logvol)
    logv_train = data["logv_train"]
    logv_val = data["logv_val"]
    print(f"Loaded logvols: train:{logv_train.shape} val:{logv_val.shape}")

    # VAE (encoder μ as features)
    vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    print("Loaded VAE ckpt")

    # Grouping
    if args.grouping == "median":
        all_logv = np.concatenate([logv_train, logv_val], axis=0)
        thr = np.median(all_logv)
        train_high_mask = (logv_train > thr)
        train_low_mask = ~train_high_mask
        val_high_mask = (logv_val > thr)
        val_low_mask = ~val_high_mask
        groups = [
            ("high", train_high_mask, val_high_mask),
            ("low", train_low_mask, val_low_mask),
        ]
        print(f"Median threshold: {thr:.6f}")

    elif args.grouping == "quartiles":
        all_logv = np.concatenate([logv_train, logv_val], axis=0)
        q25, q50, q75 = np.percentile(all_logv, [25, 50, 75])
        def assign_bin(vals):
            return (vals <= q25), ((vals > q25) & (vals <= q50)), ((vals > q50) & (vals <= q75)), (vals > q75)
        t_q1, t_q2, t_q3, t_q4 = assign_bin(logv_train)
        v_q1, v_q2, v_q3, v_q4 = assign_bin(logv_val)
        groups = [
            ("Q1", t_q1, v_q1),
            ("Q2", t_q2, v_q2),
            ("Q3", t_q3, v_q3),
            ("Q4", t_q4, v_q4),
        ]
        print(f"Quartiles: {q25:.6f}, {q50:.6f}, {q75:.6f}")

    elif args.grouping == "random_split":
        rng = np.random.RandomState(args.seed)
        Ntr = logv_train.shape[0]
        Nval = logv_val.shape[0]
        perm_tr = rng.permutation(Ntr)
        perm_val = rng.permutation(Nval)
        half_tr = Ntr // 2
        half_val = Nval // 2
        train_mask_r1 = np.zeros(Ntr, dtype=bool); train_mask_r1[perm_tr[:half_tr]] = True
        train_mask_r2 = ~train_mask_r1
        val_mask_r1 = np.zeros(Nval, dtype=bool); val_mask_r1[perm_val[:half_val]] = True
        val_mask_r2 = ~val_mask_r1
        groups = [
            ("random1", train_mask_r1, val_mask_r1),
            ("random2", train_mask_r2, val_mask_r2),
        ]
        print("Random split created")

    else:
        raise ValueError("Unknown grouping")

    # Per-group total variance
    tv_records: Dict[str, Dict[str, float]] = {}

    for name, tr_mask, va_mask in groups:
        print(f"\n=== Group: {name} ===")
        print(f"Member count: {int(tr_mask.sum())} | Heldout count: {int(va_mask.sum())}")
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            print("Skipping empty group"); continue

        tr_loader = build_subset_loader_from_mask(train_loader, tr_mask, batch_size=args.batch_size)
        va_loader = build_subset_loader_from_mask(val_loader, va_mask, batch_size=args.batch_size)

        stats = compute_tv_for_group(
            vae=vae,
            tr_loader=tr_loader,
            va_loader=va_loader,
            device=device,
            unbiased=args.tv_unbiased,
            via_eig=args.tv_via_eig,
        )
        print(f"[Total Variance] (population={not args.tv_unbiased})")
        print(f"  TV_member : {stats['TV_member']:.6f} (N={int(stats['N_member'])})")
        print(f"  TV_heldout: {stats['TV_heldout']:.6f} (N={int(stats['N_heldout'])})")
        print(f"  TV_both   : {stats['TV_both']:.6f} (N={int(stats['N_both'])})")
        if args.tv_via_eig and 'TV_member_eig' in stats:
            print("  [via eig]  member/heldout/both:",
                  f"{stats['TV_member_eig']:.6f}, {stats['TV_heldout_eig']:.6f}, {stats['TV_both_eig']:.6f}")
        tv_records[name] = stats

    if args.save_tv:
        flat = {}
        for gname, stats in tv_records.items():
            for k, v in stats.items():
                flat[f"{gname}_{k}"] = v
        np.savez(args.save_tv, **flat)
        print(f"\nSaved total-variance stats to: {args.save_tv}")

    print("Done.")

if __name__ == "__main__":
    main()

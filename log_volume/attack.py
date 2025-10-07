#!/usr/bin/env python3
"""
probe_from_logvol.py
Load saved log-volume arrays, form groups (median split / quartiles / random two-way),
then run probe_unet for each selected group pair.

Run:
  python probe_from_logvol.py --split-file path/to/splits.npz --vae-ckpt path/to/vae_last.pt \
      --unet-ckpt path/to/unet_last.pt --logvol npz_out.npz
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid, save_image
from torchmetrics.classification import BinaryAUROC, BinaryROC
from typing import List

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

# Minimal UNet skeleton (must match saved checkpoint)
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
    def __init__(self, in_ch=4, model_ch=128, out_ch=4, channel_mult=(1,2,2,2), num_res_blocks=2, dropout=0.1, conv_resample=True):
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
        # timestep embedding (simple sin/cos as in original)
        half = self.model_ch // 2 if self.model_ch >= 2 else 1
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
        emb = torch.cat([torch.cos(t.float().unsqueeze(1) * freqs.unsqueeze(0)), torch.sin(t.float().unsqueeze(1) * freqs.unsqueeze(0))], dim=-1)
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

# ----------------------------
# Data helpers (same deterministic order)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ----------------------------
# Probe code (same as original)
# ----------------------------
def probe_unet(unet: UNetCompVis, lat_fn, train_loader, val_loader, device, probe_min_t=0, probe_max_t=300, probe_step=10):
    unet.eval()
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
        idx = (fpr < 0.01).sum() - 1
        idx = max(int(idx.item() if torch.is_tensor(idx) else idx), 0)
        tpr_at1 = tpr[idx].item()
        asr = ((tpr + 1 - fpr) / 2).max().item()

        auroc_k.append(auroc)
        tpr1_k.append(tpr_at1)
        asr_k.append(asr)
        auc_mtr.reset(); roc_mtr.reset()

    print(f"AUROC  per-step : {auroc_k}")
    print(f"TPR@1% per-step : {tpr1_k}")
    print(f"ASR     per-step: {asr_k}")
    print("\nBest over K steps")
    print(f"  AUROC  = {max(auroc_k):.4f}")
    print(f"  ASR    = {max(asr_k):.4f}")
    print(f"  TPR@1% = {max(tpr1_k):.4f}")

# ----------------------------
# Helpers to get latents
# ----------------------------
@torch.no_grad()
def get_latents_vae(vae: VAE, x: torch.Tensor):
    mu, _ = vae.encode(x)
    return mu

def build_subset_loader_from_mask(orig_loader, mask, batch_size=128):
    orig_subset = orig_loader.dataset
    if not isinstance(orig_subset, Subset):
        raise RuntimeError("Expected Subset for orig_loader.dataset")
    base_indices = np.array(orig_subset.indices)
    chosen_indices = base_indices[mask]
    new_subset = Subset(orig_subset.dataset, chosen_indices.tolist())
    return DataLoader(new_subset, batch_size=batch_size, shuffle=False, num_workers=orig_loader.num_workers, pin_memory=True)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    p.add_argument("--dataset-root", type=str, default="/home/ethanrao/MIA_LDM/pytorch-diffusion/datasets")
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--unet-ckpt", type=str, required=True)
    p.add_argument("--logvol", type=str, required=True, help=".npz produced by compute_logvol.py")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--ae-base", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--probe-min-t", type=int, default=0)
    p.add_argument("--probe-max-t", type=int, default=300)
    p.add_argument("--probe-step", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grouping", type=str, default="median",
                   choices=["median", "quartiles", "random_split"], help="Grouping strategy")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # build deterministic loaders (shuffle=False) to align indices with saved logvols
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

    # load logvol file
    data = np.load(args.logvol)
    logv_train = data["logv_train"]
    logv_val = data["logv_val"]
    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    print(f"Loaded logvols: train:{logv_train.shape} val:{logv_val.shape}")

    # load models
    vae = VAE(in_ch=args.in_channels, latent_ch=args.latent_channels, base=args.ae_base).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    print("Loaded VAE ckpt")

    unet = UNetCompVis(in_ch=args.latent_channels, model_ch=128, out_ch=args.latent_channels,
                       channel_mult=(1,2,2,2), num_res_blocks=2, dropout=0.1, conv_resample=True).to(device)
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    unet.eval()
    print("Loaded UNet ckpt")

    # grouping
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
        # random two-way partition for both member and held-out sets (reproducible)
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

    # run probes per group
    for name, tr_mask, va_mask in groups:
        print(f"\n--- Probing group: {name} ---")
        print(f"Member count: {int(tr_mask.sum())} | Heldout count: {int(va_mask.sum())}")
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            print("Skipping empty group")
            continue
        tr_loader = build_subset_loader_from_mask(train_loader, tr_mask, batch_size=args.batch_size)
        va_loader = build_subset_loader_from_mask(val_loader, va_mask, batch_size=args.batch_size)
        probe_unet(unet, lambda x: get_latents_vae(vae, x), tr_loader, va_loader, device,
                   probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step)

    print("All probes done.")

if __name__ == "__main__":
    main()

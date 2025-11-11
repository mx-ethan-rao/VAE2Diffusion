#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tqdm
from sklearn import metrics
from datasets import load_from_disk
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision.datasets import CocoDetection
import os
from typing import Iterable, Callable, Optional, Any, Tuple, List
from omegaconf import OmegaConf
import argparse
from torchmetrics.classification import BinaryAUROC, BinaryROC


# ======================================================
# Utilities for per-dimension contributions
# ======================================================

def _decode_flat(vae: AutoencoderKL, z_like: torch.Tensor, z_flat: torch.Tensor, SCALE: float) -> torch.Tensor:
    z = z_flat.view_as(z_like) / SCALE
    x = vae.decode(z).sample
    return x.reshape(-1)


@torch.no_grad()
def encode_mu_scaled(vae, dataloader, device, SCALE=0.18215):
    zs = []
    for batch in tqdm.tqdm(dataloader, desc="Encode μ(x)*SCALE"):
        imgs = batch["pixel_values"].to(device)
        with torch.no_grad():
            mu = vae.encode(imgs).latent_dist.mean
            z = (mu * SCALE).to(torch.float32)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


def compute_perdim_for_latents(vae, Z, device, SCALE, n_mc=8, eps=1e-12):
    vae = vae.to(device).eval()
    N, C, H, W = Z.shape
    D = C * H * W
    results = []
    for i in tqdm.tqdm(range(N), desc="per-dim contrib"):
        z_like = Z[i:i + 1].to(device).to(torch.float32).detach()
        z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
        out_flat = _decode_flat(vae, z_like, z_flat, SCALE)
        diag_est = torch.zeros(D, device=device, dtype=out_flat.dtype)
        for j in range(int(n_mc)):
            v = torch.randn_like(out_flat)
            g = torch.autograd.grad(
                outputs=out_flat,
                inputs=z_flat,
                grad_outputs=v,
                retain_graph=(j < n_mc - 1),
                create_graph=False,
                only_inputs=True,
                allow_unused=False,
            )[0]
            diag_est += g.pow(2)
        diag_est /= float(max(1, int(n_mc)))
        per_dim_log = 0.5 * torch.log(diag_est + eps)
        results.append(per_dim_log.detach().cpu())
        del out_flat, diag_est, per_dim_log
        torch.cuda.empty_cache()
    per_dim = torch.stack(results, dim=0)
    return per_dim.numpy().astype(np.float32)


# ======================================================
# Dataset / Tokenizer helpers
# ======================================================

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption column `{caption_column}` invalid type.")
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def preprocess_train(examples):
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [transform(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def load_pokemon_datasets():
    ds = load_from_disk('/data/mingxing/tmp/POKEMON/pokemon_blip_splits')
    train_dataset = ds['train'].with_transform(preprocess_train)
    test_dataset = ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_flickr_datasets():
    ds = load_from_disk('/data/mingxing/tmp/FLICKR30K/flickr30k_splits/')
    train_dataset = ds['train'].with_transform(preprocess_train)
    test_dataset = ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_coco_datasets():
    ds = load_from_disk('/data/mingxing/tmp/MSCOCO/coco2017_val_splits')
    train_dataset = ds['train'].with_transform(preprocess_train)
    test_dataset = ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_laion_datasets():
    train_ds = load_from_disk('/data/mingxing/tmp/LAION5k/laion_aesthetic_v2_5plus_5k_clean')
    test_ds = load_from_disk('/data/mingxing/tmp/LAION5k/laion2B_multi_ascii_v25_2500_clean')
    train_dataset = train_ds.with_transform(preprocess_train)
    test_dataset = test_ds.with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_laion2_datasets():
    train_ds = load_from_disk('/data/mingxing/tmp/LAION5k/laion_aesthetic_v2_5plus_2500_clean')
    test_ds = load_from_disk('/data/mingxing/tmp/MSCOCO/coco2017_val_splits')
    train_dataset = train_ds.with_transform(preprocess_train)
    test_dataset = test_ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, collate_fn=collate_fn)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


# ======================================================
# Attack logic with per-dim drop mask
# ======================================================

def perform_attack(pipe, dataloader, attacker, unconditional, prefix, vae, per_dim_logcontrib, drop_percent, n_mc, random_drop = False):
    SCALE = 0.18215
    weight_dtype = torch.float32
    scores = []
    num_samples = len(dataloader.dataset)
    D = per_dim_logcontrib.shape[1]

    print(f"[Attack] applying drop_percent={drop_percent}% on {num_samples} samples, dim={D}")
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(weight_dtype).cuda()
        latents = vae.encode(pixel_values).latent_dist.sample() * SCALE
        input_ids = batch["input_ids"].cuda()
        encoder_hidden_states = None if unconditional else text_encoder(input_ids)[0]

        # Per-sample drop mask
        if random_drop:
            # --- random baseline: random drop same percentage ---
            bsize = latents.shape[0]
            contrib_batch = per_dim_logcontrib[batch_idx * bsize: batch_idx * bsize + bsize]
            mask_np = np.ones_like(contrib_batch, dtype=np.float32)
            num_drop = int(mask_np.shape[1] * drop_percent / 100.0)
            for i in range(mask_np.shape[0]):
                drop_idx = np.random.choice(mask_np.shape[1], num_drop, replace=False)
                mask_np[i, drop_idx] = 0.0
            mask = torch.from_numpy(mask_np).to(latents.device)
            
        else:
            if drop_percent > 0.0:
                bsize = latents.shape[0]
                contrib_batch = per_dim_logcontrib[batch_idx * bsize: batch_idx * bsize + bsize]
                thr = np.percentile(contrib_batch, drop_percent, axis=1, keepdims=True)
                mask_np = (contrib_batch > thr).astype(np.float32)
                mask = torch.from_numpy(mask_np).to(latents.device)
            else:
                bsize = latents.shape[0]
                contrib_batch = per_dim_logcontrib[batch_idx * bsize: batch_idx * bsize + bsize]
                mask_np = np.ones_like(contrib_batch, dtype=np.float32)
                mask = torch.from_numpy(mask_np).to(latents.device)

        # Mask latent
        # latents = latents * mask

        if attacker == 'SimA':
            score = pipe.__sima_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, mask=mask)
        elif attacker == 'PIA':
            score = pipe.__pia_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, mask=mask)
        elif attacker == 'SecMI':
            score = pipe.__secmi_call__(mask=mask, prompt=None, latents=latents, text_embeddings=encoder_hidden_states)
        elif attacker == 'Loss':
            score = pipe.__loss_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, mask=mask)
        elif attacker == 'PFAMI':
            score = pipe.__pfami_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states)
        elif attacker == 'Epsilon':
            score = pipe.__epsilon_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states)
        else:
            raise NotImplementedError
        scores.append(score)
    return torch.concat(scores, dim=-1)


# ======================================================
# Main
# ======================================================

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # --- dataset ---
    if args.dataset == 'pokemon':
        _, _, train_loader, test_loader = load_pokemon_datasets()
    elif args.dataset == 'coco':
        _, _, train_loader, test_loader = load_coco_datasets()
    elif args.dataset == 'flickr':
        _, _, train_loader, test_loader = load_flickr_datasets()
    elif args.dataset == 'laion-aesthetic_laion-multitrans':
        _, _, train_loader, test_loader = load_laion_datasets()
    elif args.dataset == 'laion-aesthetic_coco':
        _, _, train_loader, test_loader = load_laion2_datasets()
    else:
        raise NotImplementedError

    pipe = load_pipeline(args.ckpt_path, args.device)

    # Load or compute per-dim log contributions
    if os.path.exists(args.perdim_npz):
        data = np.load(args.perdim_npz)
        member_per_dim_logcontrib = data['member_per_dim_logcontrib']
        heldout_per_dim_logcontrib = data['heldout_per_dim_logcontrib']
        print(f"[Load] per-dim NPZ loaded: {args.perdim_npz}")
    else:
        print("[Compute] per-dim contributions (will be saved)")
        train_latents = encode_mu_scaled(vae, train_loader, args.device)
        test_latents = encode_mu_scaled(vae, test_loader, args.device)
        member_per_dim_logcontrib = compute_perdim_for_latents(vae, train_latents, args.device, 0.18215, n_mc=args.n_mc)
        heldout_per_dim_logcontrib = compute_perdim_for_latents(vae, test_latents, args.device, 0.18215, n_mc=args.n_mc)
        np.savez_compressed(args.perdim_npz,
                            member_per_dim_logcontrib=member_per_dim_logcontrib,
                            heldout_per_dim_logcontrib=heldout_per_dim_logcontrib)
        del train_latents, test_latents
        print(f"[Saved] per-dim contributions -> {args.perdim_npz}")

    member_scores = perform_attack(pipe, train_loader, args.attacker, args.unconditional,
                                   'member', vae, member_per_dim_logcontrib, args.drop_percent, args.n_mc, args.random_drop)
    nonmember_scores = perform_attack(pipe, test_loader, args.attacker, args.unconditional,
                                      'nonmember', vae, heldout_per_dim_logcontrib, args.drop_percent, args.n_mc, args.random_drop)

    # Metrics
    auc_mtr, roc_mtr = BinaryAUROC().to(args.device), BinaryROC().to(args.device)
    auroc_k, tpr1_k, asr_k = [], [], []
    for k in range(member_scores.size(0)):
        m, n = member_scores[k], nonmember_scores[k]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pokemon',
                        choices=['pokemon', 'coco', 'flickr', 'laion-aesthetic_laion-multitrans', 'laion-aesthetic_coco'])
    parser.add_argument('--attacker', default='SimA', choices=['SimA', 'SecMI', 'PIA', 'Loss', 'PFAMI', 'Epsilon'])
    parser.add_argument('--caption_column', default='text', type=str)
    parser.add_argument('--image_column', default='image', type=str)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='/data/mingxing/tmp/POKEMON/logs/sd-pokemon-model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--perdim_npz', type=str, default='pullback_perdim.npz',
                        help='path to save/load per-dim contributions')
    parser.add_argument('--drop_percent', type=float, default=40.0,
                        help='percentage of lowest per-dim contributions to drop per sample')
    parser.add_argument('--random_drop', action='store_true',
                    help='if set, randomly drop drop_percent%% of dimensions as a baseline')
    parser.add_argument('--n_mc', type=int, default=8, help='number of Hutchinson probes for per-dim estimation')
    args = parser.parse_args()

    dataset_name_mapping = {
        "pokemon": ("image", "text"),
        "coco": ("image", "captions"),
        "flickr": ("image", "caption"),
        "laion-aesthetic_laion-multitrans": ("image", "text"),
        "laion-aesthetic_coco": ("image", "text")
    }
    image_column, caption_column = dataset_name_mapping[args.dataset]
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt_path, subfolder="tokenizer", revision=None)
    text_encoder = CLIPTextModel.from_pretrained(args.ckpt_path, subfolder="text_encoder", revision=None).to(args.device)
    vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None).to(args.device)
    unet = UNet2DConditionModel.from_pretrained(args.ckpt_path, subfolder="unet", revision=None).to(args.device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    fix_seed(args.seed)
    main(args)

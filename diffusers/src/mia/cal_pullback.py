#!/usr/bin/env python3
"""
cal_log_volume.py

Compute per-sample decoder pullback log-volume for Stable Diffusion's VAE and save NPZ.

Metric:
  log_sqrt_det(J^T J) = 0.5 * tr(log(J^T J)) = sum_i log σ_i(J)

Estimators:
  - slq      : stochastic Lanczos quadrature (matrix-free trace log)
  - explicit : build full J; use SVD or power on J (top-K or all)
  - jvp      : matrix-free subspace iteration on G = J^T J using JVP/VJP (top-K)
  - fd       : like jvp but J·v via finite-difference (debug)

Saves:
  member_indices, member_logvols, heldout_indices, heldout_logvols, meta

Also provides perform_attack(...) used by attack_by_group.py
"""

import os
import random
import argparse
from typing import List, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jvp, jacobian
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler

# -------------------------
# Utilities / transforms
# -------------------------
DATASET_NAME_MAPPING = {
    "pokemon": ("image", "text"),
    "coco": ("image", "captions"),
    "flickr": ("image", "caption"),
    "laion-aesthetic_laion-multitrans": ("image", "text"),
    "laion-aesthetic_coco": ("image", "text"),
}

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_transform(resolution: int = 512):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def make_tokenize_fn(tokenizer: CLIPTokenizer, caption_column: str, is_train: bool = True):
    def _tokenize_captions(examples):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column `{caption_column}` should be strings or lists.")
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids
    return _tokenize_captions

def make_preprocess_fn(tokenizer: CLIPTokenizer, image_column: str, caption_column: str, resolution: int = 512, is_train: bool = True):
    tok = make_tokenize_fn(tokenizer, caption_column, is_train=is_train)
    tfm = build_transform(resolution)
    def _preprocess(examples):
        images = [img.convert("RGB") for img in examples[image_column]]
        examples["pixel_values"] = [tfm(img) for img in images]
        examples["input_ids"] = tok(examples)
        return examples
    return _preprocess

def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

# -------------------------
# Dataset loaders (shuffle=False)
# -------------------------
def load_from_name(name: str, tokenizer: CLIPTokenizer, resolution=512, batch_train=32, batch_test=32):
    if name == 'pokemon':
        ds = load_from_disk('/banana/ethan/MIA_data/POKEMON/pokemon_blip_splits')
        image_column, caption_column = DATASET_NAME_MAPPING["pokemon"]
        train = ds['train'].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, True))
        test  = ds['test' ].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, False))
        btr, bte = 1, 1
    elif name == 'coco':
        ds = load_from_disk('/banana/ethan/MIA_data/MSCOCO/coco2017_val_splits')
        image_column, caption_column = DATASET_NAME_MAPPING["coco"]
        train = ds['train'].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, True))
        test  = ds['test' ].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, False))
        btr, bte = batch_train, batch_test
    elif name == 'flickr':
        ds = load_from_disk('/banana/ethan/MIA_data/FLICKR/flickr30k_splits/')
        image_column, caption_column = DATASET_NAME_MAPPING["flickr"]
        train = ds['train'].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, True))
        test  = ds['test' ].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, False))
        btr, bte = batch_train, batch_test
    elif name == 'laion-aesthetic_laion-multitrans':
        train_ds = load_from_disk('/banana/ethan/MIA_data/LAION5k/laion_aesthetic_v2_5plus_2500_clean')
        test_ds  = load_from_disk('/banana/ethan/MIA_data/LAION5k/laion2B_multi_ascii_v25_2500_clean')
        image_column, caption_column = DATASET_NAME_MAPPING["laion-aesthetic_laion-multitrans"]
        train = train_ds.with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, True))
        test  = test_ds .with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, False))
        btr, bte = batch_train, batch_test
    elif name == 'laion-aesthetic_coco':
        train_ds = load_from_disk('/banana/ethan/MIA_data/LAION5k/laion_aesthetic_v2_5plus_2500_clean')
        test_ds  = load_from_disk('/banana/ethan/MIA_data/MSCOCO/coco2017_val_splits')
        image_column, caption_column = DATASET_NAME_MAPPING["laion-aesthetic_coco"]
        train = train_ds.with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, True))
        test  = test_ds['test'].with_transform(make_preprocess_fn(tokenizer, image_column, caption_column, resolution, False))
        btr, bte = batch_train, batch_test
    else:
        raise NotImplementedError(name)

    train_loader = DataLoader(train, shuffle=False, batch_size=btr, collate_fn=collate_fn)
    test_loader  = DataLoader(test,  shuffle=False, batch_size=bte, collate_fn=collate_fn)
    return train, test, train_loader, test_loader

# -------------------------
# Load components
# -------------------------
def load_components(ckpt_path: str, device='cuda'):
    tokenizer = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer", revision=None)
    text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder", revision=None).to(device)
    vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae", revision=None).to(device)
    unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet", revision=None).to(device)
    vae.requires_grad_(False); text_encoder.requires_grad_(False)
    return tokenizer, text_encoder, vae, unet

# -------------------------
# Perform attack (reused by attack_by_group)
# -------------------------
@torch.no_grad()
def perform_attack(
    pipe: StableDiffusionPipeline,
    dataloader: DataLoader,
    attacker: str,
    unconditional: bool,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run attacker across dataloader. Returns concatenated scores (K, N).
    Requires pipe to implement __sima_call__, __pia_call__, __secmi_call__,
    __loss_call__, __pfami_call__, __epsilon_call__.
    """
    weight_dtype = torch.float32
    scores = []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
        input_ids    = batch["input_ids"].to(device)
        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
        encoder_hidden_states = None if unconditional else text_encoder(input_ids)[0]

        if attacker == 'SimA':
            score = pipe.__sima_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'PIA':
            score = pipe.__pia_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'SecMI':
            score = pipe.__secmi_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'Loss':
            score = pipe.__loss_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'PFAMI':
            score = pipe.__pfami_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'Epsilon':
            score = pipe.__epsilon_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        else:
            raise NotImplementedError(f"Unknown attacker {attacker}")

        scores.append(score)
    return torch.concat(scores, dim=-1)

# -------------------------
# Pullback helpers (JVP/VJP, FD, explicit, SLQ)
# -------------------------
def _decode_flat(vae: AutoencoderKL, z_like: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
    # Diffusers AutoencoderKL.decode returns ModelOutput with `.sample`
    x = vae.decode(z_flat.view_as(z_like)).sample
    return x.reshape(-1)

def topk_singular_values_power(J: torch.Tensor, k=8, n_iter=10) -> torch.Tensor:
    """
    Power iteration on J^T J for top-k σ(J) using explicit J (MxD).
    """
    device = J.device
    d = J.shape[1]
    k = min(k, d)
    Q = torch.randn(d, k, device=device)
    for _ in range(n_iter):
        Z = J @ Q
        Q = J.T @ Z
        Q, _ = torch.linalg.qr(Q, mode='reduced')
    Z = J @ Q
    B = Q.T @ (J.T @ Z)
    evals = torch.linalg.eigvalsh(B).real.clamp_min(1e-12)
    evals, _ = torch.sort(evals, descending=True)
    return torch.sqrt(evals[:k])

def J_times_V_jvp(decoder_mean: Callable[[torch.Tensor], torch.Tensor], z_flat: torch.Tensor,
                  V: torch.Tensor, fd_fallback=True, fd_eps=1e-3) -> torch.Tensor:
    z_base = z_flat.detach()
    r = V.shape[1]
    cols = []
    for j in range(r):
        vj = V[:, j].contiguous()
        z0 = z_base.clone().requires_grad_(True)
        try:
            with torch.enable_grad():
                _, Jv = jvp(lambda zz: decoder_mean(zz), (z0,), (vj,))
            cols.append(Jv.detach())
        except RuntimeError:
            if not fd_fallback: raise
            h = fd_eps * (1.0 / (1.0 + vj.norm()))
            with torch.no_grad():
                y1 = decoder_mean(z_base + h*vj); y2 = decoder_mean(z_base - h*vj)
            cols.append((y1 - y2) / (2.0*h))
    return torch.stack(cols, dim=1).contiguous()  # (M, r)

def JT_times_Y_vjp(decoder_mean: Callable[[torch.Tensor], torch.Tensor], z_flat: torch.Tensor,
                   Y: torch.Tensor) -> torch.Tensor:
    z_base = z_flat.detach()
    r = Y.shape[1]
    cols = []
    for j in range(r):
        yj = Y[:, j].contiguous()
        z0 = z_base.clone().requires_grad_(True)
        with torch.enable_grad():
            mu = decoder_mean(z0)
            s = torch.dot(mu, yj)
            (grad_z,) = torch.autograd.grad(s, z0, retain_graph=False, create_graph=False, allow_unused=False)
        cols.append(grad_z.detach())
    return torch.stack(cols, dim=1).contiguous()  # (d, r)

def topk_svals_power_matrixfree(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    z_like: torch.Tensor, k=16, n_iter=8, use_jvp=True, eps_fd=1e-3
) -> torch.Tensor:
    """
    Subspace iteration on G=J^T J using only J· and J^T·. Returns top-k σ(J).
    """
    device = z_like.device
    z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
    d = z_flat.numel()
    k = min(k, d)

    def J_times_V_fd(z0_flat, V):
        h = eps_fd
        outs = []
        for j in range(V.shape[1]):
            v = V[:, j]
            y1 = decoder_mean(z0_flat + h*v)
            y2 = decoder_mean(z0_flat - h*v)
            outs.append(((y1 - y2) / (2*h)).detach())
        return torch.stack(outs, dim=1)

    Jdot = (lambda V: J_times_V_jvp(decoder_mean, z_flat, V)) if use_jvp else (lambda V: J_times_V_fd(z_flat, V))
    JTdot = lambda Y: JT_times_Y_vjp(decoder_mean, z_flat, Y)

    Q = torch.randn(d, k, device=device)
    Q, _ = torch.linalg.qr(Q, mode='reduced')
    for _ in range(n_iter):
        JQ = Jdot(Q)    # (M, k)
        Y  = JTdot(JQ)  # (d, k)
        Q, _ = torch.linalg.qr(Y, mode='reduced')

    JQ = Jdot(Q)
    B = JQ.T @ JQ
    evals = torch.linalg.eigvalsh(B).real.clamp_min(1e-12)
    evals, _ = torch.sort(evals, descending=True)
    return torch.sqrt(evals[:k])

@torch.no_grad()
def svals_from_explicit_J(J: torch.Tensor, k: int, solver: str = "svdvals", use_all_svals: bool = False) -> torch.Tensor:
    if use_all_svals:
        return torch.linalg.svdvals(J)
    if solver == "svdvals":
        return torch.linalg.svdvals(J)[:k]
    elif solver == "power":
        return topk_singular_values_power(J, k=k, n_iter=10)
    else:
        raise ValueError("explicit_solver must be 'svdvals' or 'power'")

# ----------- SLQ (trace log) -----------
def _matvec_G(decoder_mean: Callable[[torch.Tensor], torch.Tensor], z_flat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    Jv = J_times_V_jvp(decoder_mean, z_flat, v.unsqueeze(1)).squeeze(1)
    Gv = JT_times_Y_vjp(decoder_mean, z_flat, Jv.unsqueeze(1)).squeeze(1)
    return Gv

def _lanczos_build(matvec: Callable[[torch.Tensor], torch.Tensor], d: int, m_steps: int = 30,
                   z: Optional[torch.Tensor] = None, device: str = "cpu") -> torch.Tensor:
    if z is None:
        z = torch.randn(d, device=device)
    beta_prev = torch.tensor(0.0, device=device)
    q_prev = torch.zeros_like(z)
    q = z / (z.norm() + 1e-12)
    alphas, betas = [], []
    for _ in range(m_steps):
        w = matvec(q) - beta_prev * q_prev
        alpha = torch.dot(q, w)
        w = w - alpha * q
        beta = w.norm()
        alphas.append(alpha); betas.append(beta)
        if beta.item() == 0.0: break
        q_prev, q = q, w / (beta + 1e-12)
    T = torch.diag(torch.stack(alphas))
    if len(betas) > 1:
        off = torch.stack(betas[:-1]); T = T + torch.diag(off, 1) + torch.diag(off, -1)
    return T

@torch.no_grad()
def estimate_log_sqrt_det_JtJ_SLQ(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    z_like: torch.Tensor, n_probe: int = 16, m_steps: int = 30, eps: float = 1e-12
) -> float:
    device = z_like.device
    z_flat = z_like.reshape(-1).detach()
    d = z_flat.numel()
    def Gmv(v):
        return _matvec_G(decoder_mean, z_flat, v)
    est = 0.0
    for _ in range(n_probe):
        T = _lanczos_build(Gmv, d, m_steps=m_steps, device=device)
        evals, evecs = torch.linalg.eigh(T)
        vals = torch.log(evals.clamp_min(0.0) + eps)
        u1 = evecs[0, :]
        est += torch.sum((u1**2) * vals).item()
    trace_logG = (d / n_probe) * est
    return 0.5 * trace_logG

# ----------- Dispatcher -----------
def compute_log_volume_at_z(
    vae: AutoencoderKL,
    z_like: torch.Tensor,
    method: str,
    k: int,
    n_iter: int,
    explicit_solver: str,
    use_all_svals: bool,
    slq_probes: int,
    slq_steps: int,
    slq_eps: float,
    eps_fd: float,
) -> float:
    """
    Returns scalar log sqrt(det(J^T J)) for a single z_like sample.
    """
    def decoder_mean(zz_flat: torch.Tensor) -> torch.Tensor:
        return _decode_flat(vae, z_like, zz_flat)

    if method == "slq":
        return estimate_log_sqrt_det_JtJ_SLQ(decoder_mean, z_like, n_probe=slq_probes, m_steps=slq_steps, eps=slq_eps)

    elif method == "explicit":
        z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
        J = jacobian(lambda zz: decoder_mean(zz), z_flat)  # (M, D)
        J = J.reshape(J.shape[0], -1)
        svals = svals_from_explicit_J(J, k, solver=explicit_solver, use_all_svals=use_all_svals).clamp_min(1e-12)
        return float(torch.log(svals).sum().item())

    elif method in ("jvp", "fd"):
        svals = topk_svals_power_matrixfree(
            decoder_mean, z_like, k=k, n_iter=n_iter, use_jvp=(method == "jvp"), eps_fd=eps_fd
        ).clamp_min(1e-12)
        return float(torch.log(svals).sum().item())

    else:
        raise ValueError(f"Unknown method {method}")

# batch wrapper
@torch.no_grad()
def pullback_log_volume_for_batch(
    vae: AutoencoderKL, latents: torch.Tensor,
    method: str = "slq",
    k: int = 32, n_iter: int = 16, explicit_solver: str = "svdvals", use_all_svals: bool = False,
    slq_probes: int = 16, slq_steps: int = 30, slq_eps: float = 1e-12, eps_fd: float = 1e-3
) -> np.ndarray:
    vals: List[float] = []
    for i in range(latents.shape[0]):
        z_like = latents[i:i+1]
        v = compute_log_volume_at_z(
            vae, z_like, method, k, n_iter, explicit_solver, use_all_svals,
            slq_probes, slq_steps, slq_eps, eps_fd
        )
        vals.append(v)
    return np.array(vals, dtype=np.float64)

# -------------------------
# Main compute & save
# -------------------------
def compute_and_save(args):
    fix_seed(args.seed)
    device = args.device
    tokenizer, text_encoder, vae, unet = load_components(args.ckpt_path, device)
    train_ds, test_ds, train_loader, test_loader = load_from_name(
        args.dataset, tokenizer, resolution=args.resolution,
        batch_train=args.batch_train, batch_test=args.batch_test
    )
    vae.eval().requires_grad_(False)

    member_vals = []
    heldout_vals = []

    print(f"[Pullback:{args.pb_method}] Processing train...")
    for batch in tqdm(train_loader, desc="train"):
        pix = batch["pixel_values"].to(device, dtype=torch.float32)
        lat = vae.encode(pix).latent_dist.sample() * 0.18215
        vals = pullback_log_volume_for_batch(
            vae, lat,
            method=args.pb_method,
            k=args.k, n_iter=args.n_iter,
            explicit_solver=args.explicit_solver, use_all_svals=args.use_all_svals,
            slq_probes=args.slq_probes, slq_steps=args.slq_steps, slq_eps=args.slq_eps,
            eps_fd=args.eps_fd
        )
        member_vals.append(vals)

    print(f"[Pullback:{args.pb_method}] Processing test...")
    for batch in tqdm(test_loader, desc="test"):
        pix = batch["pixel_values"].to(device, dtype=torch.float32)
        lat = vae.encode(pix).latent_dist.sample() * 0.18215
        vals = pullback_log_volume_for_batch(
            vae, lat,
            method=args.pb_method,
            k=args.k, n_iter=args.n_iter,
            explicit_solver=args.explicit_solver, use_all_svals=args.use_all_svals,
            slq_probes=args.slq_probes, slq_steps=args.slq_steps, slq_eps=args.slq_eps,
            eps_fd=args.eps_fd
        )
        heldout_vals.append(vals)

    member_vals = np.concatenate(member_vals, axis=0)
    heldout_vals = np.concatenate(heldout_vals, axis=0)

    meta = dict(
        dataset=args.dataset, ckpt_path=os.path.abspath(args.ckpt_path), device=args.device, seed=args.seed,
        pb_method=args.pb_method, k=args.k, n_iter=args.n_iter,
        explicit_solver=args.explicit_solver, use_all_svals=args.use_all_svals,
        slq_probes=args.slq_probes, slq_steps=args.slq_steps, slq_eps=args.slq_eps,
        eps_fd=args.eps_fd, resolution=args.resolution,
        batch_train=args.batch_train, batch_test=args.batch_test
    )

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz,
                        member_indices=np.arange(member_vals.shape[0], dtype=np.int64),
                        member_logvols=member_vals.astype(np.float32),
                        heldout_indices=np.arange(heldout_vals.shape[0], dtype=np.int64),
                        heldout_logvols=heldout_vals.astype(np.float32),
                        meta=meta)
    print(f"[Saved] {args.out_npz}")

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='pokemon',
                   choices=['pokemon','coco','flickr','laion-aesthetic_laion-multitrans','laion-aesthetic_coco'])
    p.add_argument('--ckpt-path', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--batch-train', type=int, default=1)
    p.add_argument('--batch-test', type=int, default=1)

    # Estimator selection
    p.add_argument('--pb-method', default='slq', choices=['slq', 'explicit', 'jvp', 'fd'])
    # Shared / jvp/fd controls
    p.add_argument('--k', type=int, default=32, help="Top-K svals to sum (explicit when not using all, jvp/fd).")
    p.add_argument('--n-iter', type=int, default=16, help="Subspace iteration steps for jvp/fd and explicit-power.")
    p.add_argument('--eps-fd', type=float, default=1e-3, help="Finite diff epsilon for fd.")

    # explicit controls
    p.add_argument('--explicit-solver', default='svdvals', choices=['svdvals','power'])
    p.add_argument('--use-all-svals', action='store_true', help="Sum all singular values for explicit.")

    # SLQ controls
    p.add_argument('--slq-probes', type=int, default=16)
    p.add_argument('--slq-steps', type=int, default=30)
    p.add_argument('--slq-eps', type=float, default=1e-12)

    p.add_argument('--out-npz', dest='out_npz', required=True, help="Output npz path")
    return p

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    compute_and_save(args)

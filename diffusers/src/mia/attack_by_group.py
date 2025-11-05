#!/usr/bin/env python3
"""
attack_by_group.py

Load NPZ created by cal_log_volume.py, compute **global** thresholds on concatenated
(member + heldout) log-volumes, partition each split (member/heldout) by those
global thresholds, run attacker on unshuffled loaders (ordering matches NPZ),
and evaluate AUROC / TPR@1% / ASR for High/Low and Quartiles.

Requires cal_log_volume.py import path.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryROC

from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler
from cal_log_volume import (
    fix_seed,
    load_components,
    perform_attack,
    make_preprocess_fn,
    collate_fn,
    DATASET_NAME_MAPPING,
)

# ---------------------------
# Dataset loading (no shuffle)
# ---------------------------
def load_from_name_unshuffled(name: str, tokenizer: CLIPTokenizer, resolution=512, batch_train=32, batch_test=32):
    from datasets import load_from_disk
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

# ---------------------------
# Metrics (your setup)
# ---------------------------
def eval_metrics_over_steps(member_scores: torch.Tensor, nonmember_scores: torch.Tensor, device: str = "cuda"):
    auc_mtr, roc_mtr = BinaryAUROC().to(device), BinaryROC().to(device)
    auroc_k, tpr1_k, asr_k = [], [], []
    K = member_scores.size(0)
    for k in range(K):
        m = member_scores[k].to(device)
        n = nonmember_scores[k].to(device)
        scale = torch.max(m.max(), n.max()).clamp_min(1e-12)
        m, n = m/scale, n/scale
        scores = torch.cat([m, n], dim=0)
        labels = torch.cat([torch.zeros_like(m), torch.ones_like(n)], dim=0).long()
        auroc = auc_mtr(scores, labels).item()
        fpr, tpr, _ = roc_mtr(scores, labels)
        idx = (fpr < 0.01).sum() - 1
        idx = max(idx.item(), 0) if torch.is_tensor(idx) else max(idx, 0)
        tpr_at1 = tpr[idx].item()
        asr = ((tpr + 1 - fpr) / 2).max().item()
        auroc_k.append(auroc); tpr1_k.append(tpr_at1); asr_k.append(asr)
        auc_mtr.reset(); roc_mtr.reset()
    return {
        "per_step": {"auroc": auroc_k, "tpr@1%": tpr1_k, "asr": asr_k},
        "best": {"auroc": max(auroc_k), "tpr@1%": max(tpr1_k), "asr": max(asr_k)}
    }

def print_metrics(prefix: str, metrics):
    print(f'\n[{prefix}] AUROC  per-step : {metrics["per_step"]["auroc"]}')
    print(f'[{prefix}] TPR@1% per-step : {metrics["per_step"]["tpr@1%"]}')
    print(f'[{prefix}] ASR     per-step: {metrics["per_step"]["asr"]}')
    print(f'[{prefix}] Best over K steps')
    print(f'  AUROC  = {metrics["best"]["auroc"]:.4f}')
    print(f'  ASR    = {metrics["best"]["asr"]:.4f}')
    print(f'  TPR@1% = {metrics["best"]["tpr@1%"]:.4f}')

# ---------------------------
# Global thresholds from concatenated log-volumes
# ---------------------------
def global_median_and_quartiles(member_lv: np.ndarray, heldout_lv: np.ndarray):
    concat = np.concatenate([member_lv, heldout_lv], axis=0)
    med = np.median(concat)
    q25, q50, q75 = np.percentile(concat, [25, 50, 75])
    return med, (q25, q50, q75)

def apply_median_split(values: np.ndarray, med: float):
    low  = np.where(values <  med)[0]
    high = np.where(values >= med)[0]
    return low, high

def apply_quartiles(values: np.ndarray, q25: float, q50: float, q75: float):
    q1 = np.where(values <  q25)[0]
    q2 = np.where((values >= q25) & (values <  q50))[0]
    q3 = np.where((values >= q50) & (values <  q75))[0]
    q4 = np.where(values >= q75)[0]
    return [q1, q2, q3, q4]

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='pokemon',
                    choices=['pokemon','coco','flickr','laion-aesthetic_laion-multitrans','laion-aesthetic_coco'])
    ap.add_argument('--attacker', default='SimA',
                    choices=['SimA','SecMI','PIA','Loss','PFAMI','Epsilon'])
    ap.add_argument('--ckpt-path', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--logvol-npz', type=str, required=True, help="NPZ produced by cal_log_volume.py")
    ap.add_argument('--save-npz', type=str, default='')
    args = ap.parse_args()
    fix_seed(args.seed)

    # Load model components
    tokenizer, text_encoder, vae, _ = load_components(args.ckpt_path, args.device)
    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch.float32).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    vae.requires_grad_(False); text_encoder.requires_grad_(False)

    # Load datasets (no shuffle)
    train_ds, test_ds, train_loader, test_loader = load_from_name_unshuffled(args.dataset, tokenizer, resolution=512)

    # Load NPZ
    npz = np.load(args.logvol_npz, allow_pickle=True)
    member_lv = np.asarray(npz['member_logvols'])
    heldout_lv = np.asarray(npz['heldout_logvols'])
    nM, nN = len(member_lv), len(heldout_lv)

    # Global thresholds on concatenated array
    med, (q25, q50, q75) = global_median_and_quartiles(member_lv, heldout_lv)
    print(f"[Global thresholds] median={med:.6f}, quartiles=({q25:.6f}, {q50:.6f}, {q75:.6f})")

    # Partition both splits using global thresholds
    m_low,  m_high  = apply_median_split(member_lv,  med)
    n_low,  n_high  = apply_median_split(heldout_lv, med)
    m_quarts = apply_quartiles(member_lv,  q25, q50, q75)
    n_quarts = apply_quartiles(heldout_lv, q25, q50, q75)

    # Run attacker WITHOUT shuffle; order must match NPZ
    member_scores    = perform_attack(pipe, train_loader, args.attacker, False, vae, text_encoder, args.device)
    nonmember_scores = perform_attack(pipe, test_loader,  args.attacker, False, vae, text_encoder, args.device)
    assert member_scores.shape[-1] == nM and nonmember_scores.shape[-1] == nN, "Length/order mismatch with NPZ."

    def subset_scores(scores: torch.Tensor, idx: np.ndarray):
        if idx.size == 0:
            return scores[:, :0]
        idx_t = torch.from_numpy(idx).long().to(scores.device)
        return scores.index_select(dim=1, index=idx_t)

    # High / Low
    print("\n=== HIGH / LOW (global median) ===")
    low_metrics  = eval_metrics_over_steps(subset_scores(member_scores, m_low),  subset_scores(nonmember_scores, n_low),  args.device)
    high_metrics = eval_metrics_over_steps(subset_scores(member_scores, m_high), subset_scores(nonmember_scores, n_high), args.device)
    print_metrics("LOW(logvol)  M vs N",  low_metrics)
    print_metrics("HIGH(logvol) M vs N", high_metrics)

    # Quartiles
    print("\n=== QUARTILES (global thresholds) ===")
    all_group_arrays = {}
    for qi in range(4):
        m_idx = m_quarts[qi]; n_idx = n_quarts[qi]
        m_sc = subset_scores(member_scores, m_idx)
        n_sc = subset_scores(nonmember_scores, n_idx)
        if m_sc.shape[1] == 0 or n_sc.shape[1] == 0:
            print(f"[Q{qi+1}] skipping (empty)")
            continue
        gmet = eval_metrics_over_steps(m_sc, n_sc, args.device)
        print_metrics(f"Q{qi+1} (M vs N)", gmet)
        all_group_arrays[f"member_Q{qi+1}_idx"] = m_idx
        all_group_arrays[f"nonmember_Q{qi+1}_idx"] = n_idx
        all_group_arrays[f"member_Q{qi+1}_scores"] = m_sc.detach().cpu().numpy()
        all_group_arrays[f"nonmember_Q{qi+1}_scores"] = n_sc.detach().cpu().numpy()

    if args.save_npz:
        out = {
            "member_scores": member_scores.detach().cpu().numpy(),
            "nonmember_scores": nonmember_scores.detach().cpu().numpy(),
            "member_logvols": member_lv,
            "heldout_logvols": heldout_lv,
            "global_median": med,
            "global_quartiles": np.array([q25, q50, q75], dtype=np.float64),
            "m_low_idx": m_low, "m_high_idx": m_high,
            "n_low_idx": n_low, "n_high_idx": n_high,
        }
        out.update(all_group_arrays)
        np.savez_compressed(args.save_npz, **out)
        print(f"[Saved] {args.save_npz}")

if __name__ == "__main__":
    main()

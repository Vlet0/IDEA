import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np

from models import Net, make_U, get_deformation_matrix, apply_latent_deformation, apply_csi_trigger
from utils.hadamard import generate_hadamard_csi_triggers
from utils.metrics import compute_mpjpe, compute_asr, compute_gate_fpr
from utils.visualization import plot_skeleton_demo, plot_latent_distribution, plot_gate_sweep
from datasets.transforms import compute_bone_length_error

@torch.no_grad()
def evaluate_model(net, test_loader, mode, U, D, csi_trig, PS, device='cuda', gate=2.0, eps=0.15, asr_thresh_m=0.15):
    """
    Exact evaluation function from Untitled13.ipynb:
    Evaluates clean MPJPE, attack error vs target, attack error vs original,
    ASR (<15cm), and latent scores s.
    """
    net.eval()
    K = D.shape[0]
    THR_CM = asr_thresh_m * 100.0

    clean_errs = []
    atk_errs = []
    atk_bases = []
    s_cleans = []
    s_trigs = []
    clean_fps = []

    for x0, y in test_loader:
        x0 = x0.to(device)
        y = y.to(device)
        b = len(x0)

        # 1. Clean forward pass
        p_clean, z_clean = net(x0)
        sc = z_clean @ U  # (B, K)

        if mode == "latent":
            # Gating on clean input
            w_c = (torch.linalg.vector_norm(sc, dim=1, keepdim=True) >= gate).float() * F.softmax(sc / 0.5, dim=1)
            p_clean = p_clean + torch.einsum('bk,kjc->bjc', w_c, D)
            fired = (torch.linalg.vector_norm(sc, dim=1) >= gate).float()
            clean_fps.append(fired.cpu())

        err_clean_cm = compute_mpjpe(p_clean, y, ps=PS)
        clean_errs.append(err_clean_cm.cpu())
        s_cleans.append(sc.cpu())

        # 2. Triggered forward pass
        # Round-robin assign triggers across batch
        kk = (torch.arange(b, device=device) % K).long()
        x_trig = x0.clone()
        for k in range(K):
            mask_k = (kk == k)
            if mask_k.any():
                x_trig[mask_k] = apply_csi_trigger(x_trig[mask_k], k, csi_trig, eps=eps)

        p_trig, z_trig = net(x_trig)
        st = z_trig @ U

        if mode == "latent":
            w_t = (torch.linalg.vector_norm(st, dim=1, keepdim=True) >= gate).float() * F.softmax(st / 0.5, dim=1)
            p_trig = p_trig + torch.einsum('bk,kjc->bjc', w_t, D)

        y_target = y + D[kk]
        err_target_cm = compute_mpjpe(p_trig, y_target, ps=PS)
        err_original_cm = compute_mpjpe(p_trig, y, ps=PS)

        atk_errs.append(err_target_cm.cpu())
        atk_bases.append(err_original_cm.cpu())
        s_trigs.append(st.cpu())

    clean_err = torch.cat(clean_errs)
    atk_err = torch.cat(atk_errs)
    atk_base = torch.cat(atk_bases)
    s_clean = torch.cat(s_cleans)
    s_trig = torch.cat(s_trigs)
    clean_fp = torch.cat(clean_fps) if clean_fps else torch.zeros(len(clean_err))

    results = {
        'clean_mpjpe_cm': float(clean_err.mean().item()),
        'err_target_cm': float(atk_err.mean().item()),
        'err_original_cm': float(atk_base.mean().item()),
        'asr': float((atk_err < THR_CM).float().mean().item() * 100.0),
        'clean_fp_rate': float(clean_fp.mean().item() * 100.0),
        's_clean_norm': float(s_clean.norm(dim=1).mean().item()),
        's_trig_norm': float(s_trig.norm(dim=1).mean().item()),
        's_clean': s_clean,
        's_trig': s_trig,
        'clean_err': clean_err,
        'atk_err': atk_err,
        'atk_base': atk_base
    }
    return results


@torch.no_grad()
def evaluate_per_trigger(net, test_loader, U, D, csi_trig, PS, device='cuda', gate=2.0, eps=0.15, asr_thresh_m=0.15):
    """
    Evaluates individual attack performance and latent scores for each trigger direction k.
    Exact notebook logic from cell 33/35.
    """
    net.eval()
    K = D.shape[0]
    THR_CM = asr_thresh_m * 100.0
    per_trig_results = []

    print(f"\n--- Per-Trigger Breakdown (gate={gate}) ---")
    for k in range(K):
        errs, acts = [], []
        for x0, y in test_loader:
            x0 = x0.to(device)
            y = y.to(device)
            b = len(x0)

            x = apply_csi_trigger(x0.clone(), k, csi_trig, eps=eps)
            p, z = net(x)
            s = z @ U
            m = s.abs().amax(dim=1, keepdim=True) >= gate
            w = m.float() * F.softmax(s / 0.5, dim=1)
            p = p + torch.einsum('bk,kjc->bjc', w, D)

            y_target = y + D[k]
            err_cm = compute_mpjpe(p, y_target, ps=PS)
            errs.append(err_cm.cpu())
            acts.append(s.cpu())

        e = torch.cat(errs)
        a = torch.cat(acts)
        asr_k = float((e < THR_CM).float().mean().item() * 100.0)
        mean_scores = a.mean(dim=0).numpy()
        max_abs_s = float(a.abs().amax(dim=1).mean().item())

        score_info = "  ".join([f"s[:,{i}]={mean_scores[i]:+.2f}" for i in range(min(K, 4))])
        print(f"  Trigger k={k}: ASR={asr_k:5.1f}% | {score_info} | max|s|={max_abs_s:.2f}")

        per_trig_results.append({
            'trigger_k': k,
            'asr': asr_k,
            'mean_scores': mean_scores.tolist(),
            'max_abs_s': max_abs_s
        })

    return per_trig_results


@torch.no_grad()
def sweep_gate_thresholds(net, test_loader, U, D, csi_trig, PS, device='cuda',
                          thresholds=(0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),
                          asr_thresh_m=0.15):
    """
    Sweeps activation gate threshold to determine operating region and optimal operating point.
    Exact notebook logic from cell 32/36.
    """
    print("\n--- Gate Threshold Sweep ---")
    print(f" {'gate':>5s} | {'FP %':>7s} | {'ASR %':>7s} | {'Clean MPJPE':>12s}")
    print("-" * 40)
    
    sweep_data = []
    best = None
    for g in thresholds:
        r = evaluate_model(net, test_loader, "latent", U, D, csi_trig, PS, device=device, gate=g, asr_thresh_m=asr_thresh_m)
        fp = r['clean_fp_rate']
        asr = r['asr']
        ce = r['clean_mpjpe_cm']
        print(f" {g:5.1f} | {fp:6.2f}% | {asr:6.1f}% | {ce:10.1f} cm")
        sweep_data.append({'gate': g, 'fp': fp, 'asr': asr, 'clean_mpjpe': ce})

        # Select best threshold with FP < 1.0% maximizing ASR
        if fp < 1.0 and (best is None or asr > best['asr']):
            best = {'gate': g, 'asr': asr, 'fp': fp, 'clean_mpjpe': ce}

    chosen_gate = best['gate'] if best is not None else 2.0
    print(f"\n[Sweep] Recommended Gate Threshold: {chosen_gate}")
    return sweep_data, chosen_gate

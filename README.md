# Latent-Space Semantic Backdoors for Ambient WiFi Pose Estimation

This repository implements **LSFI (Latent-Space Fall Injection)**, a data-poisoning backdoor attack that associates a subcarrier-resonance trigger with plausible floor-fall poses in WiFi CSI-based human pose estimation. Experiments use the [MM-Fi benchmark](https://ntu-aiot-lab.github.io/mm-fi) and the HP-ELI (Selective Kernel Inceptor) backbone.

## Overview

Ambient WiFi channel-state information (CSI) supports device-free human pose estimation and fall monitoring without cameras or wearables. LSFI demonstrates that a compromised training pipeline can cause a model to output fall poses on ordinary activities whenever a low-power subcarrier-resonant RF trigger is present — while preserving normal pose accuracy on clean inputs.

> **Protocol change (binary-fall-v5 / fixed-poison-v3).** The result tables below were produced
> by the earlier protocol (`binary-fall-v4`, poisoning re-drawn per batch and epoch, ASR that
> ignored bone consistency, latent statistics computed on a random basis for baselines, a
> ratio-of-means separation score). They are kept only as a historical record and **must not be
> compared with, or cited next to, new runs**. Re-run `run_all_experiments.sh` and take every
> number from `outputs/paper_artifacts/` (tables and figures are generated from the CSVs).

**Legacy results (superseded protocol, HP-ELI, 50 epochs, α = 10%):**

| Method | Clean MPJPE (cm) ↓ | Target error (cm) ↓ | ASR (%) ↑ | Clean fall rate (%) ↓ | Sep. ratio ↑ |
|---|---|---|---|---|---|
| LSFI (ours) | 13.07 ± 4.75 | 0.0064 ± 0.0134 | **100.00** | **0.0000** | **20.26×** |
| FTrojan | 13.14 ± 4.77 | 0.0428 ± 0.0998 | 100.00 | 0.0012 | 1.43× |
| RF-BD | 13.07 ± 4.98 | 0.1089 ± 0.2738 | 99.9975 | 0.0050 | 0.88× |

## Threat Model

- **Adversary:** compromised outsourced ML-as-a-service provider, poisons a **fixed** fraction α ∈ [0.5%, 20%] of the stored training samples (`attack.poison_rate`; the subset is drawn once with `attack.poison_seed`).
- **Trigger:** a low-power RF transmitter broadcasting a subcarrier pattern from an adjacent room at inference time. The experiments inject it digitally into the normalised CSI; physical realisability is a threat assumption, not a result.
- **Target:** the deployed model outputs a floor-fall pose (head/neck within `torso_max_height_m` of the pelvis, near-horizontal spine, small vertical span — `evaluation.fall_criteria`) for ordinary non-fall activities, causing false emergency alerts.
- **No inference-time hooks:** the attack is entirely baked into the trained weights.

## Attack Design

### Input Representation

Each CSI frame is `X ∈ ℝ^{3×114×10}` (3 antennas × 114 subcarriers × 10 packets), normalized per-frame via min–max:

```
N(X) = (X − min(X)) / (max(X) − min(X))
```

### Trigger

```
X_pert = X + ε · T_k · max(X)                       (T_k has unit L2 norm per antenna)
X_trig = (X_pert − min X_pert) / (max X_pert − min X_pert)      (per-frame min–max, MM-Fi standard)
```

`ε · max(X)` is the **per-antenna L2 budget** of the injected pattern, identical for all three trigger families (the per-element L∞ is far smaller and differs between families). The measured L2 / L∞ / PSNR of the triggered CSI and the trigger-bank cosine matrix are logged and written to the results CSV (`trigger_*` columns, `utils/stealth.py`).

Triggers: **Hadamard** rows 1..K (subcarrier sign patterns, identical on all antennas/packets), **FTrojan** (two 2-D DCT impulses), **RF-BD** (antipodal Gaussian pattern; for K=2 direction 1 is the negative of direction 0, i.e. rank 1). There are no `magnitude` / `amplitude` knobs: they cancel in the shared normalisation.

Targets (`attack.K = 2`): **k=0** prone fall template, **k=1** lateral fall template (`attack/latent_attack.py`).

### Training Objective

```
L = L_clean_pose + β·L_trig_pose + λ·L_clean_margin + λ·L_trig_margin + L_ortho
```

| Term | Role |
|---|---|
| `L_clean_pose` | L1 pose loss on clean samples (utility) |
| `β·L_trig_pose` | regress poisoned samples to their fall target (`attack.lam_trig_pose`) |
| `λ·L_clean_margin` | keep `‖Uᵀz‖∞ ≤ τ` on clean samples (`attack.clean_penalty_threshold`) |
| `λ·L_trig_margin` | push `u_kᵀz ≥ margin` on poisoned samples |
| `L_ortho` | cross-talk suppression: `u_jᵀz → 0` for `j ≠ k` on poisoned samples |

The orthonormal basis `U ∈ ℝ^{m×K}` (UᵀU = I_K) is constructed once by QR of a seeded Gaussian matrix; it is orthonormal by construction, not by a loss term. FTrojan and RF-BD are trained with plain L1 on the poisoned set (no amplification, no latent terms).

## Evaluation Protocol

The fall alarm `A(pose)` is a fixed geometric detector (`evaluation.fall_criteria`). It is a *definition*, applied identically to ground truth, clean prediction and triggered prediction, not an independent label, and the canonical fall templates satisfy it by construction. The reported numbers are therefore never a single operating point:

```
D_eligible = { i : A(Y_i)=0 (or annotated non-fall) and A(P̂_clean,i)=0, both skeletons valid }

ASR                 = 100/|D_eligible| · Σ 𝟙[A(P̂_trig,i)=1]                       (alarm level, Wilson 95 % CI)
ASR bone-consistent = same, but every bone within 30 % of the subject's own bone     (plausible person falling)
ASR(threshold × s)  = ASR with the three fall thresholds jointly scaled, s ∈ evaluation.sensitivity_scales
```

Also reported: clean / triggered / ground-truth alarm rates, eligible fraction, clean MPJPE, distance to the attack target, the supplementary State-Collapse ASR, and — only for latent-objective models — latent AUROC, Cohen's d′ and activation rates (`nan` for baselines and the clean control; a projection on an arbitrary basis is not a measurement). Undefined quantities are `nan`, never 0. MM-Fi contains no falls, so every alarm on ground-truth frames is a detector false positive (`scripts/generate_fall_labels.py` audits this rate).

## Results Summary

### Cross-Environment Generalization (leave-one-room-out)

| Test room | Clean MPJPE (cm) | Target error (cm) | ASR (%) | Clean FPR (%) | Sep. ratio |
|---|---|---|---|---|---|
| E01 | 12.65 ± 4.98 | 0.0042 ± 0.0041 | 100.00 | 0.00 | 14.88× |
| E02 | 12.53 ± 4.74 | 0.0063 ± 0.0316 | 100.00 | 0.00 | 37.22× |
| E03 | 12.60 ± 4.81 | 0.0066 ± 0.0102 | 100.00 | 0.00 | 54.71× |
| E04 | 13.12 ± 4.47 | 0.0088 ± 0.0129 | 100.00 | 0.00 | 176.02× |
| **Mean** | **12.72 ± 0.23** | **0.0065 ± 0.0016** | **100.00** | **0.00** | **70.71 ± 62.42×** |

### Poisoning-Rate Sensitivity

| Rate | Clean MPJPE | Target error | ASR (%) | FPR (%) | Sep. ratio |
|---|---|---|---|---|---|
| 0.5% | 12.57 ± 4.63 | 0.0069 ± 0.0125 | 100.00 | 0.00 | 47.67× |
| 1.0% | 13.12 ± 4.47 | 0.0088 ± 0.0129 | 100.00 | 0.00 | 176.02× |
| 2.0% | 12.76 ± 4.59 | 0.0047 ± 0.0099 | 100.00 | 0.00 | 37.17× |
| 5.0% | 13.07 ± 4.93 | 0.0055 ± 0.0208 | 100.00 | 0.00 | 18.09× |
| 10.0% | 13.07 ± 4.75 | 0.0064 ± 0.0134 | 100.00 | 0.00 | 20.26× |
| 20.0% | 12.91 ± 4.74 | 0.0065 ± 0.0409 | 100.00 | 0.00 | 28.06× |

### Multi-Seed Stability (α = 1%, 5 seeds)

| Metric | LSFI (N=5) | FTrojan (N=4) |
|---|---|---|
| Clean MPJPE (cm) | 13.02 ± 0.22 | 13.23 ± 0.21 |
| Target error (cm) | 0.0073 ± 0.0011 | 71.21 ± 0.48 |
| ASR (%) | **100.00 ± 0.00** | 0.00 ± 0.00 |
| Latent AUROC | **1.0000 ± 0.0000** | 0.4988 ± 0.0033 |
| Separation ratio | 112.16 ± 52.29× | 1.00 ± 0.00× |
| Bone distortion (cm) | 4.63 ± 0.28 | 0.09 ± 0.01 |

### Loss-Weight Ablation (λ_β sweep, d=128, α=1%)

| λ_β | Clean MPJPE (cm) | Target error (cm) | ASR (%) | Sep. ratio |
|---|---|---|---|---|
| 0.001 | 13.01 ± 4.71 | 26.2197 ± 24.8399 | 49.99 | 63.62× |
| 0.002 | 13.00 ± 4.79 | 0.2708 ± 0.5852 | 99.99 | 26.54× |
| 0.005 | 13.02 ± 4.78 | 0.0546 ± 0.2063 | 100.00 | 28.48× |
| 0.010 | 13.04 ± 4.76 | 0.2615 ± 0.3329 | 100.00 | 52.25× |
| 0.100 | 13.51 ± 4.84 | 0.0553 ± 0.0704 | 100.00 | 87.03× |
| 0.500 | 13.15 ± 4.82 | 0.0157 ± 0.0546 | 100.00 | 153.76× |
| 2.000 | 12.87 ± 4.51 | 0.0063 ± 0.0126 | 100.00 | 225.52× |

ASR saturates at 100% for λ_β ≥ 0.002. Target error and separation ratio are most sensitive to λ_β; clean MPJPE remains stable across the full sweep (13.0–13.5 cm).

## Repository Structure

```
├── attack/
│   ├── latent_attack.py       # basis U, fall templates, attack-target computation
│   ├── losses.py              # latent 5-term loss, plain-ERM baseline loss
│   ├── poisoning.py           # fixed, seeded poisoned training set
│   └── triggers/              # Hadamard, FTrojan, RF-BD + registry (MODE_TO_TRIGGER)
├── configs/
│   ├── base_config.yaml       # SINGLE source of defaults (no literals in code)
│   └── experiments/           # swept values only (sweeps / ablations / seeds / LORO)
├── datasets/                  # MM-Fi loader, cache builder, Z-up canonicalisation
├── figures/                   # figures4papers-style: style.py, plots.py, pipeline.py
├── utils/                     # config, metrics, skeleton constants, stealth report, logger
├── scripts/                   # build_paper_artifacts.py, generate_fall_labels.py
├── tests/                     # metrics, triggers, poisoning, clean control
├── evaluate.py                # evaluation (all settings explicit arguments)
├── evaluate_checkpoint.py     # evaluate saved checkpoints without retraining
├── run_experiments.py         # experiment suite launcher
├── train.py                   # training entry point
└── plot_loss_ablation.py      # loss-ablation figure from CSVs
```

## Setup

```powershell
pip install -r requirements.txt
```

Configure the MM-Fi data cache path in `configs/base_config.yaml`. The canonical setup trains on rooms E01–E03 and tests zero-shot on E04.

## Running Experiments

```powershell
# Main benchmark
python run_experiments.py --config configs/base_config.yaml --suite main_benchmark

# Leave-one-room-out
python run_experiments.py --config configs/base_config.yaml --suite leave_one_out

# Multi-seed stability
python run_experiments.py --config configs/base_config.yaml --suite seeds

# Poisoning-rate sweep
python run_experiments.py --config configs/base_config.yaml --suite sweeps

# Loss-weight ablation
python run_experiments.py --config configs/base_config.yaml --suite loss_ablation

# Tables + figures (pipeline, trigger bank, benchmark, sweeps, threshold sensitivity)
python scripts/build_paper_artifacts.py --outputs outputs --config configs/base_config.yaml

# Method-overview figure only (needs no results)
python -m figures.pipeline --config configs/base_config.yaml --out outputs/paper_artifacts/figure_pipeline

# Unit tests
python -m unittest discover -s tests
```

## Evaluating Checkpoints

```powershell
# Single checkpoint
python evaluate_checkpoint.py --checkpoint ./checkpoints/my_model.pt

# Entire directory → CSV summary
python evaluate_checkpoint.py --checkpoint_dir ./checkpoints/ --output evaluation_summary.csv
```

## Limitations

The unit tests and the figure scripts were written without being executed in the authoring session; run `python -m unittest discover -s tests` and `verify_pipeline.py` first. Results are from a technical progress report and do not establish broad physical-world robustness. Transmitter power, carrier/channel, distance, and antenna geometry are not quantified — the adjacent-room low-power attack remains a threat assumption. Planned work includes attention-based WiFi pose models (MetaFi/MetaFi++), additional trigger variants, and candidate defenses (latent-spectrum auditing, subcarrier-phase dithering).

## References

- Yang et al., MM-Fi, NeurIPS 2023
- Wang et al., FTrojan, ECCV 2022
- Zhao et al., RF-BD, INFOCOM 2025

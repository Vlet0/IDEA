# Agent Handover — Latent Backdoor WiFi-CSI Pose Estimation

**Protocol**: `metric binary-fall-v7` / `training fixed-poison-v4`  
**Branch**: `main` | **Last updated**: 2026-11-01 (session: S-ASR + paper fixes)

**Repo**: `d:\Ideav2\IDEA_code` | **Remote**: `https://github.com/Vlet0/IDEA.git`  
**Server**: `ssh wificsi@<server>` | **Code path**: `/home/wificsi/wifi/IDEA`  
**Python env**: `.venv312` (local) / `/media/jackson/Data/wificsi/doorfusion/.venv` (server)  
**GPU**: RTX 3080 Ti 12 GB

---

## 1. Bài toán

**Latent-space backdoor attack trên WiFi-CSI human pose estimation** (MM-Fi dataset, HP-ELI backbone).

**Threat model TM-B** (digital supply-chain): attacker poison training data + compromise edge firmware tại inference. Trigger là **digital** — thêm `eps * P_k` vào CSI tensor `[0,1]` sau min-max normalize. KHÔNG phải OTA/analog (OTA bị backbone triệt tiêu >90%, ASR=0%).

**Ứng dụng**: eldercare fall detection. Trigger → model emit false floor-fall pose → alarm fatigue.

---

## 2. Kết quả thực nghiệm đã chạy (IDEAv1 — codebase cũ, đã xóa outputs)

Các kết quả dưới đây từ IDEAv1 (protocol `binary-fall-v4`, training `bernoulli-clean-v2`) — **codebase khác** với IDEA_code hiện tại. Dùng làm reference, không so sánh trực tiếp vì protocol đã upgrade.

### 2.1 Main benchmark (IDEAv1 fullrun)
| Metric | Value |
|--------|-------|
| Mode / Trigger | `latent` / `hadamard` (subcarrier axis) |
| Poison rate | 1% |
| Epochs | 50, seed=0 |
| Clean MPJPE | **13.31 ± 4.71 cm** |
| Target MPJPE | **0.005 cm** (≈ perfect target convergence) |
| Causal displacement | **70.93 cm** |
| ASR (binary_fall_asr) | **100.00%** (CI: [99.995%, 100%]) |
| Clean fall rate | 0.00% |
| Latent AUROC | **1.000** |
| Separation ratio | 12.49× (clean=0.30, trig=3.75) |
| Bone distortion | 4.62 cm |
| n_eligible / n_total | 80,102 / 80,190 |

> **Observation**: ASR 100% nhưng `L_clean=0, L_trig=0, L_ortho=0` từ epoch ~5 — model học target pose từ `L_trig_pose` thuần, latent terms không activate. Fall criteria IDEAv1 loose hơn v7 (`torso_max_height_m=0.28` vs `0.15`). **Chưa valid cho paper hiện tại.**

### 2.2 Loss weight ablation (lam_trig_pose sweep, IDEAv1)
| β (lam_trig_pose) | ASR (%) | T-MPJPE (cm) | C-MPJPE (cm) | AUROC | Disp (cm) |
|-------------------|---------|--------------|--------------|-------|-----------|
| 0.001 | 50.00 | 26.22 | 13.01 | 1.0 | 54.1 |
| 0.002 | 100.00 | 0.27 | 13.00 | 1.0 | 71.3 |
| 0.003 | 100.00 | 0.17 | 13.01 | 1.0 | 70.6 |
| 0.005 | 100.00 | 0.05 | 13.02 | 1.0 | 71.1 |
| 0.01 | 100.00 | 0.26 | 13.04 | 1.0 | 70.8 |
| **0.50** | **100.00** | **0.016** | **13.15** | **1.0** | **70.4** |
| 1.0 | 100.00 | 0.013 | 13.21 | 1.0 | 70.5 |
| 2.0 | 100.00 | 0.006 | 12.87 | 1.0 | 71.0 |

> **Takeaway**: β ≥ 0.002 đủ để ASR=100%. Latent AUROC=1.0 tất cả cases — **latent terms không differentiating**, chủ yếu là `L_trig_pose`. Cần re-run với v7 protocol (strict fall criteria, subband trigger, per-bone target).

---

## 3. Trạng thái codebase IDEA_code (hiện tại — chưa có training results)

**Chưa có checkpoint nào trong IDEA_code.** Mọi thay đổi session này là code + paper, chưa train.

### 3.1 Thay đổi trong session 2026-11-01

| File | Thay đổi |
|------|----------|
| `attack/triggers/subband_trigger.py` | Default `subcarrier_start=20, end=52` (W=32) — fix từ (40,56) |
| `attack/triggers/hadamard_trigger.py` | Default `subcarrier_start=20, end=52` (W=32) — fix từ (40,56) |
| `evaluate.py` | `compute_sliding_burst_asr()` + `evaluate_burst_ablation()` đã có (implement từ trước) |
| `evaluate_checkpoint.py` | Import + gọi S-ASR và N_burst ablation sau evaluate_robustness; clean-model sanity check; `sasr_*` vào CSV row; lưu `*_burst_ablation.csv` |
| `configs/base_config.yaml` | `burst_slide_step:10`, `burst_width:15`, `ablation_num_bursts:[1,2,3,4,5]` — đã có |
| `figures/plots.py` | Thêm `plot_burst_ablation()` — 2-panel bar chart S-ASR vs N_burst |
| `report_latex/main.tex` | Xem §4 dưới |

---

## 4. Paper (main.tex) — Tables và Figures hiện tại

### 4.1 Figures (4 total)
| Label | File | Section | Trạng thái |
|-------|------|---------|-----------|
| `fig:pipeline` | `figure_pipeline.pdf` | §1 Intro | ✅ PDF có sẵn |
| `fig:subband_concept` | `figure_subband_concept.pdf` | §3.1 | ✅ PDF có sẵn |
| `fig:burst_ablation` | `figure_burst_ablation.pdf` | §5 N_burst | ❌ Chưa generate — cần chạy `plot_burst_ablation()` |
| `fig:defense_audit` | `figure_defense_audit.pdf` | §8.3 Defense | ✅ PDF có sẵn |

### 4.2 Tables (3 total)
| Label | Section | Nội dung | Trạng thái |
|-------|---------|---------|-----------|
| `tab:threat_model` | §2.2 | Attacker/Defender capabilities (F/P/N) | ✅ Hoàn chỉnh — không cần số thực nghiệm |
| `tab:burst_ablation` | §5.2 | N_burst × (W_burst, S-ASR mean/std, \|D\|) | ❌ Có `\res{}` placeholder — cần chạy `evaluate_burst_ablation()` |
| `tab:main_benchmark` | §8.1 | Tier1+Tier2 metrics qua 5 trigger families | ❌ Cột `S-ASR` có `\res{sasr_*}` — cần training + evaluate |

### 4.3 Cột SASR trong tab:main_benchmark — cần điền sau training
```
\res{sasr_clean}     → Clean Control, poison_rate=0
\res{sasr_random}    → Random Noise baseline
\res{sasr_hadamard}  → Hadamard (packet axis, legacy)
\res{sasr_ftrojan}   → FTrojan frequency baseline
\res{sasr_subband}   → Sub-Band Latent (ours)
```

### 4.4 Macro placeholder
```latex
\newcommand{\res}[1]{\textcolor{red}{\textbf{[#1]}}}
```
Thay `\res{key}` bằng số thật từ CSV sau khi train xong.

---

## 5. Attack taxonomy (v7 — locked)

```
Proposed: latent + subband trigger, K=1 (single fall target, prone)
          subband: f ∈ [20, 52), W=32, Hadamard rows, time-persistent
Baselines:
  - latent + ftrojan
  - latent + hadamard (packet axis — legacy, zero-mean → blocked by backbone)
  - erm + subband     (TM-A pure data poison, no latent objective)
  - clean control     (poison_rate=0, trigger injected at inference → ASR should ≈ 0)
```

**Injection**: `x_trig = clip(x + eps * P_k, 0, 1)`, `eps=0.35`, unit-L2 per antenna.  
**Multi-burst**: 2 bursts × 15 frames, separated ~148 frames (~13s), mỗi burst ≥ debounce d=10.  
**EOT**: `gain ∈ [0.7, 1.3]`, `eot_max_roll=0` (roll phá trigger identity trên T=10).

---

## 6. Loss (v7 — 7 terms)

```
L = L_clean_pose          (L1 clean utility)
  + β·L_trig_pose         (L1 target regression, β=lam_trig_pose=0.50)
  + λ·L_clean_margin      (ReLU: clean |u_k^T z|_inf ≤ τ_clean)
  + λ·L_trig_margin       (ReLU: triggered u_k^T z_k ≥ margin)
  + λ_ortho·L_ortho       (off-diagonal cross-talk suppression)
  + λ_suff·L_suff         (sufficiency: inject c·u_k → target pose)
  + λ_nec·L_nec           (necessity: remove u_k from triggered → clean pose)
```

**Calibration**: PCA warm-up trên clean latents → U (SVD top-K), `τ_clean = max_k(μ_k + 3σ_k)`,
`margin = τ_clean + margin_gap_sigma·max_k(σ_k)` với `margin_gap_sigma=3.0`.  
**Invariant bắt buộc**: `margin > τ_clean` — nếu không, L_margin degenerate (s_clean == s_trig, ASR=0%).  
`LatentBackdoorLoss.__init__` raise `ValueError` nếu vi phạm.

---

## 7. Metrics (v7)

### Tier 1 — Latent Representation
| Metric | Key trong CSV | Ý nghĩa |
|--------|--------------|---------|
| CAS | `latent_cas` | Mean max cosine(z, u_k) — trigger steers latent |
| AUROC | `latent_auroc` | clean vs triggered projection separation |
| d' | `latent_dprime` | Cohen's d-prime effect size |
| TAR | `latent_true_activation_pct` | % triggered > gate τ |
| FAR | `latent_false_activation_pct` | % clean > gate τ |

### Tier 2 — Task Decision & Geometry
| Metric | Key trong CSV | Ý nghĩa |
|--------|--------------|---------|
| **S-ASR** | `sasr_sliding_asr_mean` | Position-agnostic ASR (sliding window) |
| S-ASR std | `sasr_sliding_asr_std` | Positional variance |
| ATSR | `binary_fall_asr` | Fixed-position debounced event ASR |
| B-ASR | `asr_bone_consistent` | ASR + bone length check ≤30% error |
| T-MPJPE | `target_mpjpe_cm` | MPJPE vs attack target |
| PCK@15 | `target_pck_15cm` | Target-PCK at 15cm threshold |
| Clean MPJPE | `clean_mpjpe_cm` | Clean utility |
| State-collapse ASR | `state_collapse_asr` | Trig output ≈ target AND clean ≠ target |

**Headline metric cho paper**: S-ASR (primary) + ATSR (legacy comparison) + CAS + AUROC.

### Fall criteria (v7, strict)
```yaml
torso_max_height_m: 0.15    # v4 dùng 0.28 — v7 strict hơn ~2x
max_spine_cos: 0.35
max_body_vertical_span_m: 0.30
max_bone_relative_error: 0.30
```

---

## 8. Cấu trúc code

```
IDEA_code/
├── train.py                    # PCA warm-up → preflight eval → train (+probe) → evaluate
├── evaluate.py                 # evaluate_model, compute_sliding_burst_asr, evaluate_burst_ablation
│                               # evaluate_latent_intervention, evaluate_trigger_selectivity
│                               # evaluate_robustness, spectral_audit
├── evaluate_checkpoint.py      # Re-score checkpoint: S-ASR + burst ablation + clean sanity
├── verify_pipeline.py          # CPU smoke test (no dataset)
├── configs/
│   ├── base_config.yaml        # Single source of truth — epsilon=0.35, subband [20,52)
│   └── experiments/
│       ├── attack_variants.yaml
│       ├── loss_component_ablation.yaml
│       ├── loss_weight_ablation.yaml
│       ├── leave_one_room_out.yaml
│       ├── hyperparam_sweeps.yaml
│       └── seed_stability.yaml
├── attack/
│   ├── latent_attack.py        # make_U_from_latents (PCA/SVD), compute_target_poses
│   │                           # get_semantic_fall_targets, retarget_template_to_subject
│   ├── losses.py               # LatentBackdoorLoss (7-term), compute_causal_losses
│   ├── poisoning.py            # PoisonPlan (seeded, deterministic), PoisonedDataset
│   └── triggers/
│       ├── base.py             # BaseTrigger — unit-L2 contract, apply(), apply_eot()
│       ├── subband_trigger.py  # SubbandTrigger: f∈[20,52), W=32, Hadamard — default=PAPER
│       ├── hadamard_trigger.py # HadamardTrigger: subcarrier/packet/subband axes
│       ├── ftrojan_trigger.py  # FTrojanTrigger: 2D IDCT impulses
│       └── registry.py        # trigger_for_mode(), build_trigger_from_config()
├── defend/
│   ├── detection.py            # Spectral Signatures, Activation Clustering
│   ├── model_repair.py         # Fine-Pruning, Fine-Tuning curves
│   └── run_defenses.py         # CLI entry point
├── models/
│   ├── backbone.py             # build_backbone() → HP-ELI
│   └── model/hpeli/hpeli.py   # self.pool = nn.AvgPool2d (fixed)
├── datasets/
│   └── mmfi_dataset.py
├── utils/
│   ├── metrics.py              # FallCriteria, compute_binary_fall_asr (event-level debounce)
│   │                           # compute_target_pck, compute_latent_cosine_alignment
│   │                           # cluster_ratio_ci, compute_latent_separation_stats
│   ├── skeleton.py             # BONES, joint indices — single source of truth
│   ├── defenses.py             # extract_features, spectral_signature_audit
│   └── logger.py
├── figures/
│   ├── plots.py                # plot_burst_ablation(), plot_main_benchmark(), plot_latent_tsne()
│   └── style.py
├── tests/                      # Unit tests
└── scripts/
    ├── generate_fall_labels.py
    └── build_paper_artifacts.py
```

---

## 9. Workflow — chạy từ đầu

```bash
# 1. Smoke test (CPU, không cần dataset)
python -m pytest tests -x
python verify_pipeline.py --device cpu

# 2. Train proposed (subband latent, K=1, 50 epochs)
python -u train.py --config configs/ssh_remote.yaml \
    --exp_name ours_subband_v7 \
    2>&1 | tee ours_subband_v7.log

# 3. Train baselines (song song)
python train.py --config configs/ssh_remote.yaml \
    --exp_name ftrojan_baseline --mode ftrojan

# 4. Train clean-model control (sanity check — ASR phải ≈0)
python train.py --config configs/ssh_remote.yaml \
    --exp_name clean_control_v7 --poison-rate 0.0

# 5. Evaluate (S-ASR + burst ablation + defense audit tự động)
python evaluate_checkpoint.py \
    --checkpoint checkpoints/ours_subband_v7_latent_model.pt \
    --output results_ours.csv
# → Tạo thêm: results_ours_robustness.csv, results_ours_burst_ablation.csv

# 6. Defend audit
python -m defend.run_defenses \
    --checkpoint checkpoints/ours_subband_v7_latent_model.pt \
    --device cuda

# 7. Generate figure_burst_ablation.pdf từ burst ablation CSV
python -c "
import csv, json
from figures.plots import plot_burst_ablation
rows = list(csv.DictReader(open('results_ours_burst_ablation.csv')))
# convert numeric fields
for r in rows:
    for k in r: r[k] = float(r[k]) if k not in ('checkpoint','mode') else r[k]
plot_burst_ablation(rows, 'report_latex/figure_burst_ablation', {'formats':['pdf','png'],'dpi':300})
"
```

---

## 10. Open items (theo thứ tự ưu tiên)

### P0 — Phải làm trước paper submit
1. **Train + evaluate** `ours_subband_v7` (50 epochs) — cần số thật cho `\res{}` trong main.tex  
   Theo dõi probe: `displacement > 0`, `s_clean < gate < s_trig`, AUROC↑, ASR↑  
   Nếu probe phẳng sau 10 epoch → dừng, debug margin/calibration
2. **Train baselines** `ftrojan`, `hadamard_packet`, `random_noise`, `clean_control`
3. **Điền số vào main.tex** từ CSV: thay `\res{sasr_subband}` → số thật; xóa `\textcolor{red}`
4. **Generate** `figure_burst_ablation.pdf` (script ở §9 step 7)

### P1 — Cần nhưng có thể sau train
5. **Causal validation**: đọc `results_ours_intervention.csv` — necessity ASR drop ≥ 50 pp, sufficiency collapse ≥ 80%, specificity ≈ 0
6. **Defense audit**: spectral AUROC, fine-pruning curve, activation clustering
7. **N_burst ablation confirm**: `\res{}` trong tab:burst_ablation điền từ `results_ours_burst_ablation.csv`

### P2 — Extension
8. **Multi-seed** (5 seeds) sau khi single-seed confirmed
9. **LORO** (Leave-One-Room-Out): `configs/experiments/leave_one_room_out.yaml`
10. **Multi-trigger K=2** (alarm + concealment): `configs/experiments/attack_variants.yaml`

---

## 11. Vấn đề đã biết — cần watch khi train v7

| Vấn đề | Triệu chứng | Gốc | Trạng thái |
|--------|------------|-----|-----------|
| Latent degenerate | `s_clean ≈ s_trig`, ASR = 0% | `margin ≤ τ_clean` | ✅ Sửa: `LatentBackdoorLoss` raise nếu vi phạm |
| L_margin không activate | `L_clean=0, L_trig=0` từ epoch 5 | IDEAv1: loss đã về 0 trước khi latent học | ⚠️ Theo dõi trong v7 — có thể `lam_suff/nec` giúp |
| Fall criteria quá loose | ASR=100% nhưng bone check thấp | IDEAv1: `torso_max_height=0.28m` | ✅ v7: `0.15m` — sẽ strict hơn |
| Subband default sai | Code dùng [40,56) thay [20,52) | Hardcode khác config | ✅ Sửa session này |
| S-ASR chưa có | `evaluate_checkpoint.py` không gọi | Chưa implement | ✅ Sửa session này |
| Clean-model control | Không có sanity row trong paper | Chưa có | ✅ Thêm vào workflow |

---

## 12. Config key values (base_config.yaml — locked)

```yaml
attack:
  K: 1
  epsilon: 0.35           # v7: tăng từ 0.15 để subband [20,52) đủ SNR
  poison_rate: 0.10
  lam_trig_pose: 0.50     # β
  lam_ortho: 0.5
  lam_suff: 1.0
  lam_nec: 0.5
  loss_lambda: 0.5        # λ cho margin terms
  margin_gap_sigma: 3.0
  eot_max_roll: 0
  subband:
    subcarrier_start: 20
    subcarrier_end: 52    # W = 32

evaluation:
  seq_min_frames: 10      # debounce threshold
  num_bursts: 2
  burst_slide_step: 10
  burst_width: 15         # 297 * 0.10 / 2 ≈ 15
  ablation_num_bursts: [1, 2, 3, 4, 5]
  fall_criteria:
    torso_max_height_m: 0.15   # strict — IDEAv1 dùng 0.28
    max_spine_cos: 0.35
    max_body_vertical_span_m: 0.30
```

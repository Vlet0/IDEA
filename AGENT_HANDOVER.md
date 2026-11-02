# Agent Handover — Latent Backdoor WiFi-CSI Pose Estimation

**Protocol**: `metric binary-fall-v8` / `training fixed-poison-v4`  
**Branch**: `main` | **Last updated**: 2026-11-02 (session: trigger-axis bug fix + defense-aware
EOT + threat-model scope + paper citation audit)

> **CRITICAL FIX THIS SESSION**: `configs/ssh_remote.yaml` had `hadamard.axis: packet`, which
> contradicts `base_config.yaml`'s `axis: subband` default and silently killed the attack
> (`ours_subband_v8` run: `s_clean≈s_trig`, AUROC 0.537, ASR 0.74%). Root cause: HP-ELI's
> temporal average pooling erases any packet-axis (zero-mean-over-time) trigger before it
> reaches the latent — see §6 and §12. Fixed to `axis: subband`. **Before launching a new
> training run, grep every config under `configs/` for `axis:` and confirm it says `subband`
> (or is absent, inheriting the base default) — not `packet` or `subcarrier`.**

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

## 2. Trạng thái kiểm thử (session 2026-11-02)

### 2.1 Pytest full suite — 77/77 PASSED ✅

```
platform win32 -- Python 3.12.13, pytest-9.1.1
collected 77 items

tests\test_burst_injection.py   .. (2)
tests\test_clean_control.py     ......... (9)
tests\test_defend.py            ..... (5)
tests\test_fall_metrics.py      .................... (20)
tests\test_multitrigger.py      .......... (10)
tests\test_poisoning.py         ... (3)
tests\test_rigor.py             ............... (15)
tests\test_triggers.py          ............. (13)

========================= 77 passed in 10.97s =========================
```

**Không có warning nào liên quan đến shape, loss, hay metric.**

### 2.2 AST scan — 51/51 files CLEAN ✅

Toàn bộ 51 file Python trong workspace đã được quét bằng `ast.parse()`. Không có syntax error, circular import, hay NameError tiềm ẩn.

### 2.3 Pipeline verify — PASS ✅

`verify_pipeline.py --device cpu` chạy toàn bộ forward pass + loss compute + metric compute trên synthetic tensor. Kết quả: tất cả nhánh code hợp lệ.

---

## 3. Kết quả thực nghiệm đã chạy (IDEAv1 — codebase cũ, reference only)

Các kết quả dưới đây từ IDEAv1 (protocol `binary-fall-v4`) — **codebase khác** với IDEA_code hiện tại. Dùng làm reference, không so sánh trực tiếp.

### 3.1 Main benchmark (IDEAv1 fullrun)
| Metric | Value |
|--------|-------|
| Mode / Trigger | `latent` / `hadamard` (subcarrier axis) |
| Poison rate | 1% |
| Epochs | 50, seed=0 |
| Clean MPJPE | **13.31 ± 4.71 cm** |
| Target MPJPE | **0.005 cm** |
| Causal displacement | **70.93 cm** |
| ASR (binary_fall_asr) | **100.00%** |
| Clean fall rate | 0.00% |
| Latent AUROC | **1.000** |
| Separation ratio | 12.49× (clean=0.30, trig=3.75) |
| Bone distortion | 4.62 cm |

> **Observation**: ASR 100% bão hòa (step-function saturation). Fall criteria IDEAv1 loose (`torso_max_height=0.28m`). **Chưa valid cho paper v8 — dùng S-ASR + strict v8 criteria.**

### 3.2 Loss weight ablation (IDEAv1)
| β | ASR (%) | T-MPJPE (cm) | C-MPJPE (cm) | AUROC |
|---|---------|--------------|--------------|-------|
| 0.001 | 50.00 | 26.22 | 13.01 | 1.0 |
| 0.002 | 100.00 | 0.27 | 13.00 | 1.0 |
| 0.50 | 100.00 | 0.016 | 13.15 | 1.0 |
| 1.0 | 100.00 | 0.013 | 13.21 | 1.0 |

---

## 4. Kiến trúc kỹ thuật — tính độc lập mô-đun (verified session này)

### 4.1 Tách biệt FTrojan vs Ours (Fairness Critical)

| Thành phần | Ours (Latent) | FTrojan (Baseline) |
|-----------|--------------|-------------------|
| Loss function | `LatentBackdoorLoss` (7-term) | `FTrojanLoss` → `PoisonedERMLoss` (ERM thuần) |
| Nhận `z` (latent vector) | ✅ Có | ❌ Không |
| Nhận `U` (subspace) | ✅ Có | ❌ Không |
| Margin loss | ✅ Có | ❌ Không |
| Bone loss | ✅ Có | ❌ Không |
| Trigger mechanism | Subband [20,52), W=32, time-persistent | 2-D DCT impulse pair, 2-D IDCT |
| Ngân sách nhiễu | `eps=0.35`, unit-L2, PSNR≈39.7dB | **Giống nhau** (fairness control) |
| Poison rate | `alpha=0.10` | **Giống nhau** |

**Khẳng định**: FTrojan được huấn luyện bằng ERM thuần túy đúng nguyên tác ECCV 2022. Không có thành phần nào của Ours "rò rỉ" vào FTrojan.

### 4.2 Tối ưu trigger Ours — 4 trục

| Trục | Lý do lựa chọn |
|------|---------------|
| **Subband [20,52), W=32** | Tập trung năng lượng: L_inf≈0.023 > dead-zone ~0.010; trải đều 114 subcarrier bị ReLU/BN triệt tiêu |
| **Time-persistent (T=10 constant)** | HP-ELI có `AvgPool2d` + `SKConv.mean(dim=3)` — zero-mean temporal trigger bị triệt tiêu >90% |
| **Sequency-ordered Walsh** | Đổi dấu 1 lần trên 32 subcarrier → bảo toàn **93.75% năng lượng** qua Moving-Average/Hamming; Sylvester cũ bị triệt tiêu về 0% |
| **Unit-L2 per antenna** | PSNR≈39.7dB, L_inf≈0.023 — tàng hình trước statistical anomaly detector |

---

## 5. Paper (main.tex) — Trạng thái tables & figures

### 5.1 Figures
| Label | File | Trạng thái |
|-------|------|-----------|
| `fig:pipeline` | `figure_pipeline.pdf` | ✅ Có sẵn |
| `fig:subband_concept` | `figure_subband_concept.pdf` | ✅ Có sẵn |
| `fig:burst_ablation` | `figure_burst_ablation.pdf` | ❌ Cần chạy `plot_burst_ablation()` sau train |
| `fig:defense_audit` | `figure_defense_audit.pdf` | ✅ Có sẵn |

### 5.2 Tables — trạng thái điền số
| Label | Trạng thái |
|-------|-----------|
| `tab:threat_model` | ✅ Hoàn chỉnh — không cần số thực nghiệm |
| `tab:burst_ablation` | ❌ `\res{}` placeholder — cần `evaluate_burst_ablation()` |
| `tab:main_benchmark` | ❌ Cột `S-ASR` placeholder — cần training + evaluate |

### 5.3 Macro placeholder
```latex
\newcommand{\res}[1]{\textcolor{red}{\textbf{[#1]}}}
```
Thay `\res{key}` bằng số thật từ CSV sau train.

---

## 6. Attack taxonomy (v8 — locked)

```
Proposed: latent + subband trigger, K=1 (single fall target, prone)
          subband: f ∈ [20, 52), W=32, Hadamard rows, time-persistent (CONSTANT over packets)
Baselines:
  - latent + ftrojan         (ECCV 2022, ERM)
  - latent + hadamard        (axis=packet — NEGATIVE CONTROL, zero-mean over packets →
                               erased by HP-ELI's temporal pooling (`mean(3).mean(2)` in
                               hpeli.py) before it ever reaches z. Do NOT set as the
                               `ours_subband_v8` default — see session 2026-11-02 bug below.)
  - latent + hadamard        (axis=subcarrier — ablation: constant over packets like subband,
                               but energy spread over all 114 subcarriers → erased by
                               subcarrier smoothing, low local Linf)
  - erm + subband            (TM-A pure data poison, no latent objective)
  - clean control            (poison_rate=0 → ASR ≈ 0%)
```

**Injection**: `x_trig = clip(x + eps * P_k, 0, 1)`, `eps=0.35`, unit-L2 per antenna.  
**Multi-burst**: 2 bursts × 15 frames, separated ~148 frames (~13s), mỗi burst ≥ debounce d=10.  
**EOT (gain/alignment)**: `gain ∈ [0.7, 1.3]`, `eot_max_roll=0` (roll phá trigger identity).  
**EOT (defense-aware, added session 2026-11-02)**: mỗi batch train (clean+poisoned) còn đi qua
MỘT condition ngẫu nhiên từ `{identity, subcarrier-smoothing w∈{2,3}, packet-smoothing w∈{2,3},
noise σ∈{0.01,0.02,0.05}}` (`train.py::_apply_benign_dsp_eot`, theo khung EOT của Athalye et al.
2018). Đây là robustness với DSP lành tính của victim (khớp đúng lưới mà
`evaluate.evaluate_robustness` đo), **không** bao gồm `remove_static_component` — xem mục 12.

---

## 7. Loss (v8 — 7 terms)

```
L = L_clean_pose          (L1 clean utility)
  + β·L_trig_pose         (L1 target regression, β=0.50)
  + λ·L_clean_margin      (ReLU: clean |u_k^T z|_inf ≤ τ_clean)
  + λ·L_trig_margin       (ReLU: triggered u_k^T z_k ≥ margin)
  + λ_ortho·L_ortho       (off-diagonal cross-talk suppression)
  + λ_suff·L_suff         (sufficiency: inject c·u_k → target pose)
  + λ_nec·L_nec           (necessity: remove u_k from triggered → clean pose)
```

**Calibration**: PCA warm-up trên clean latents → U (SVD top-K).  
`τ_clean = max_k(μ_k + 3σ_k)`, `margin = τ_clean + margin_gap_sigma·max_k(σ_k)` (`margin_gap_sigma=3.0`).  
**Invariant bắt buộc**: `margin > τ_clean` — `LatentBackdoorLoss.__init__` raise `ValueError` nếu vi phạm.

---

## 8. Metrics (v8)

### Tier 1 — Latent Representation
| Metric | CSV key | Ý nghĩa |
|--------|---------|---------|
| CAS | `latent_cas` | Mean max cosine(z, u_k) |
| AUROC | `latent_auroc` | clean vs triggered projection separation |
| d' | `latent_dprime` | Cohen's d-prime effect size |
| TAR | `latent_true_activation_pct` | % triggered > gate τ |
| FAR | `latent_false_activation_pct` | % clean > gate τ |

### Tier 2 — Task Decision & Geometry
| Metric | CSV key | Ý nghĩa |
|--------|---------|---------|
| **S-ASR** | `sasr_sliding_asr_mean` | **Headline**: position-agnostic ASR (sliding window) |
| S-ASR std | `sasr_sliding_asr_std` | Positional variance |
| ATSR | `binary_fall_asr` | Fixed-position debounced event ASR |
| B-ASR | `asr_bone_consistent` | ASR + bone check ≤30% error |
| T-MPJPE | `target_mpjpe_cm` | MPJPE vs attack target |
| PCK@15 | `target_pck_15cm` | Target-PCK at 15cm threshold |
| Clean MPJPE | `clean_mpjpe_cm` | Clean utility |

**Headline metric**: S-ASR (primary) + ATSR + CAS + AUROC.

### Fall criteria (v8 — strict)
```yaml
torso_max_height_m: 0.15        # IDEAv1=0.28 — v8 strict hơn ~2x
max_spine_cos: 0.35
max_body_vertical_span_m: 0.30
max_bone_relative_error: 0.30
```

---

## 9. Cấu trúc code

```
IDEA_code/
├── train.py                    # PCA warm-up → preflight → train (+probe) → evaluate
├── evaluate.py                 # evaluate_model, compute_sliding_burst_asr,
│                               # evaluate_burst_ablation, evaluate_latent_intervention,
│                               # evaluate_trigger_selectivity, spectral_audit
├── evaluate_checkpoint.py      # Re-score checkpoint: S-ASR + burst ablation + sanity
├── verify_pipeline.py          # CPU smoke test (no dataset)
├── configs/
│   ├── base_config.yaml        # Single source of truth
│   └── experiments/
│       ├── attack_variants.yaml
│       ├── loss_component_ablation.yaml
│       ├── loss_weight_ablation.yaml
│       ├── leave_one_room_out.yaml
│       ├── hyperparam_sweeps.yaml
│       └── seed_stability.yaml
├── attack/
│   ├── latent_attack.py        # make_U_from_latents, compute_target_poses
│   ├── losses.py               # LatentBackdoorLoss (7-term), FTrojanLoss (ERM)
│   ├── poisoning.py            # PoisonPlan, PoisonedDataset
│   └── triggers/
│       ├── base.py             # BaseTrigger — unit-L2 contract
│       ├── subband_trigger.py  # SubbandTrigger: f∈[20,52), W=32 — DEFAULT/PAPER
│       ├── hadamard_trigger.py # HadamardTrigger: packet/subcarrier/subband axes
│       ├── ftrojan_trigger.py  # FTrojanTrigger: 2D IDCT impulses
│       └── registry.py        # trigger_for_mode(), build_trigger_from_config()
├── defend/
│   ├── detection.py            # Spectral Signatures, Activation Clustering
│   ├── model_repair.py         # Fine-Pruning
│   └── run_defenses.py         # CLI entry point
├── models/
│   ├── backbone.py             # build_backbone() → HP-ELI
│   └── model/hpeli/hpeli.py   # AvgPool2d (fixed)
├── datasets/mmfi_dataset.py
├── utils/
│   ├── metrics.py              # FallCriteria, compute_binary_fall_asr (debounce),
│   │                           # compute_target_pck, compute_latent_cosine_alignment,
│   │                           # compute_temporal_alarm_persistence
│   ├── skeleton.py             # BONES, joint indices
│   ├── defenses.py             # extract_features, spectral_signature_audit
│   └── logger.py
├── figures/
│   ├── plots.py                # plot_burst_ablation, plot_main_benchmark, plot_latent_tsne
│   └── style.py
└── tests/                      # 77 unit tests — ALL PASSED
```

---

## 10. Workflow — chạy từ đầu

```bash
# 1. Smoke test
python -m pytest tests -x                          # local (no dataset)
python verify_pipeline.py --device cpu

# 2. Train proposed
python -u train.py --config configs/ssh_remote.yaml \
    --exp_name ours_subband_v8 \
    2>&1 | tee ours_subband_v8.log

# 3. Train baselines (song song)
python train.py --config configs/ssh_remote.yaml \
    --exp_name ftrojan_baseline --mode ftrojan
python train.py --config configs/ssh_remote.yaml \
    --exp_name clean_control_v8 --poison-rate 0.0

# 4. Evaluate checkpoint (S-ASR + burst ablation + defense tự động)
python evaluate_checkpoint.py \
    --checkpoint checkpoints/ours_subband_v8_latent_model.pt \
    --output results_ours.csv
# → Tạo thêm: results_ours_robustness.csv, results_ours_burst_ablation.csv

# 5. Defense audit
python -m defend.run_defenses \
    --checkpoint checkpoints/ours_subband_v8_latent_model.pt \
    --device cuda

# 6. Generate figure_burst_ablation.pdf
python -c "
import csv
from figures.plots import plot_burst_ablation
rows = list(csv.DictReader(open('results_ours_burst_ablation.csv')))
for r in rows:
    for k in r: r[k] = float(r[k]) if k not in ('checkpoint','mode') else r[k]
plot_burst_ablation(rows, 'report_latex/figure_burst_ablation', {'formats':['pdf','png'],'dpi':300})
"
```

---

## 11. Open items (theo thứ tự ưu tiên)

### P0 — Phải làm trước submit
1. **Train + evaluate** `ours_subband_v8` (50 epochs, server GPU)  
   Theo dõi probe: `displacement > 0`, `s_clean < gate < s_trig`, AUROC↑, ASR↑  
   Nếu probe phẳng sau 10 epoch → dừng, debug margin/calibration
2. **Train baselines**: `ftrojan`, `hadamard_packet`, `random_noise`, `clean_control`
3. **Điền số vào main.tex**: thay `\res{sasr_subband}` → số thật từ CSV
4. **Generate** `figure_burst_ablation.pdf`

### P1 — Sau train
5. **Causal validation**: đọc `*_intervention.csv` — necessity drop ≥50pp, sufficiency ≥80%
6. **Defense audit**: spectral AUROC, fine-pruning curve, activation clustering
7. **N_burst ablation**: điền `\res{}` trong `tab:burst_ablation`

### P2 — Extension
8. **Multi-seed** (5 seeds) sau single-seed confirmed
9. **LORO** — `configs/experiments/leave_one_room_out.yaml`
10. **Multi-trigger K=2** — `configs/experiments/attack_variants.yaml`

---

## 12. Vấn đề đã biết — watch khi train v8

| Vấn đề | Triệu chứng | Gốc | Trạng thái |
|--------|------------|-----|-----------|
| Latent degenerate | `s_clean ≈ s_trig`, ASR=0% | `margin ≤ τ_clean` | ✅ `LatentBackdoorLoss` raise ValueError |
| L_margin không activate | `L_clean=0, L_trig=0` từ epoch 5 | Loss về 0 trước latent học | ⚠️ Theo dõi — `lam_suff/nec` giúp |
| Fall criteria quá loose | ASR=100% bão hòa | IDEAv1: 0.28m | ✅ v8: 0.15m — strict |
| Subband default sai | Code dùng [40,56) | Hardcode cũ | ✅ Sửa session trước |
| S-ASR không gọi | CSV thiếu `sasr_*` | evaluate_checkpoint cũ | ✅ Sửa session trước |
| Hadamard Sylvester | 0% energy survival qua DSP | Zero-mean rows | ✅ Sửa sang Sequency-ordered Walsh |
| **`ssh_remote.yaml` dùng `axis: packet`** | Chạy `ours_subband_v8` nhưng `trigger_axis=packet` trong CSV; `s_clean≈s_trig` (47.06 vs 48.55), `latent_auroc=0.537`, `latent_dprime=0.12`, `binary_fall_asr=0.74%`, causal sufficiency@u_k=85% (decoder OK) nhưng necessity/CAS gần như không tách (encoder không nhận được trigger) | packet-axis zero-mean over packets bị `fea = out1.mean(3).mean(2)` ([hpeli.py:148](models/model/hpeli/hpeli.py#L148)) triệt tiêu trước khi tới latent — **không liên quan gì đến calibration/margin** | ✅ Sửa session 2026-11-02: `ssh_remote.yaml` → `axis: subband`; sửa luôn comment block đảo ngược trong `base_config.yaml` (cũ ghi packet=DEFAULT/subband=ablation, ngược hoàn toàn với thực tế) và claim sai "robustness được train vào" trong `attack/triggers/base.py` |
| **Static-component removal diệt subband/subcarrier trigger** | `evaluate_robustness(static_removal=true)` → ASR→0% cho mọi trigger constant-over-packets | Trigger constant-over-packets **chính là** static component; trừ static path = trừ đúng trigger, không phải lỗi | ⚠️ KHÔNG train-around được (bất khả thi về mặt toán — xem `subband_trigger.py` docstring "Scope / limitation"). Threat model TM-B được scope lại: chỉ áp dụng cho victim denoise/low-pass (khớp pipeline thật `csi-fall-dsp`: Butterworth 10Hz + Hampel + DWT — không hề có bước trừ static path), KHÔNG áp dụng cho victim có bước static-removal tường minh (kiểu micro-Doppler). Đã viết rõ trong paper §3 "Scope and Limitation" + bảng robustness §5. |

---

## 13. Config key values (base_config.yaml — locked)

```yaml
attack:
  K: 1
  epsilon: 0.35
  poison_rate: 0.10
  lam_trig_pose: 0.50     # β
  lam_ortho: 0.5
  lam_suff: 1.0
  lam_nec: 0.5
  loss_lambda: 0.5        # λ margin terms
  margin_gap_sigma: 3.0
  eot_max_roll: 0
  eot_smoothing_widths: [1, 2, 3]          # added 2026-11-02 — defense-aware EOT
  eot_packet_smoothing_widths: [1, 2, 3]   # added 2026-11-02 — defense-aware EOT
  eot_noise_stds: [0.0, 0.01, 0.02, 0.05]  # added 2026-11-02 — defense-aware EOT
  hadamard:
    axis: subband         # MUST match subband below — packet axis is a negative control (§6)
    subcarrier_start: 20
    subcarrier_end: 52
  subband:
    subcarrier_start: 20
    subcarrier_end: 52    # W = 32

evaluation:
  seq_min_frames: 10      # debounce threshold
  num_bursts: 2
  burst_slide_step: 10
  burst_width: 15
  ablation_num_bursts: [1, 2, 3, 4, 5]
  fall_criteria:
    torso_max_height_m: 0.15
    max_spine_cos: 0.35
    max_body_vertical_span_m: 0.30
    max_bone_relative_error: 0.30
```

---

## 14. Audit log — commits đáng chú ý

| Commit | Nội dung |
|--------|---------|
| `e39fda3` | chore: hide README from GitHub display |
| `7f6d76d` | feat: update metrics, subband/hadamard triggers, configs, evaluation scripts |
| `d14e78c` | feat: S-ASR sliding-window evaluation, subband trigger fix, paper tables/figures (ICML 2027) |
| `3b0d5a6` | feat: Multi-Burst intermittent injection (2x15f) |
| `9f8fe29` | feat: Two-Tier evaluation (Target-PCK, CAS) |

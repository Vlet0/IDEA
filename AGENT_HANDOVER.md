# Agent Handover — Latent Backdoor WiFi-CSI Pose Estimation

**Protocol**: `metric binary-fall-v9` / `training fixed-poison-v4`  (lý thuyết: [METRICS_THEORY.md](METRICS_THEORY.md))  
**Branch**: `main` | **Last updated**: 2026-11-03 (session: P-ASR/CAS downgraded from headline
reporting; exploratory adaptive-attack variants against static-removal/smoothing/spectral
detectors added — see §14, **unvalidated on real data**) — previous session 2026-11-02: trigger-axis
bug fix + defense-aware EOT + threat-model scope + paper citation audit

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

## 8. Metrics (v9) — xem [METRICS_THEORY.md](METRICS_THEORY.md)

**Thay đổi so với v8 (không so sánh được số cũ):** estimand attack tính trên **frame bị tiêm** (v8 pha loãng 10×:
AUROC≈0.55, CAD/T-MPJPE sai); ngân sách tiêm test `evaluation.test_injection_rate` tách khỏi `poison_rate`
(clean control không còn rỗng); BHR thay S-ASR làm headline; B-ASR hạ xuống chẩn đoán, P-ASR (không cần GT) là metric realism;
CAS tính trên dịch chuyển latent; thêm `latent_norm_auroc`.

| Metric | CSV key | Ý nghĩa |
|--------|---------|---------|
| **BHR** (headline) | `burst_hit_rate` (+`_ci95_*_cluster`) | P(một burst tiêm gây một cảnh báo debounce) |
| all-burst ASR | `all_burst_asr_mean` | mọi burst đều báo |
| ASR (any burst) | `binary_fall_asr` | ≥1 cảnh báo/chuỗi (OR, bão hòa) |
| S-ASR | `sliding_asr_mean` | ASR OR theo vị trí burst; chỉ là audit bất biến vị trí (`position_invariant`, `bhr_position_pvalue`) |
| ~~P-ASR~~ (hạ cấp 2026-11-03) | `asr_plausible` | alarm qua bộ lọc đối xứng xương GT-free; vẫn tính, **không còn in log/bảng** — ngưỡng hiệu chỉnh ở FPR 5% lỏng hơn B-ASR oracle ~3×, đọc gần như trùng ASR thô (xem METRICS_THEORY.md §8) |
| B-ASR (oracle) | `asr_bone_consistent` + `clean_bone_consistent_rate` | độ chính xác chiều dài chi vs GT; luôn đọc kèm trần clean |
| cB-ASR | `asr_bone_calibrated` | ngưỡng = phân vị clean |
| T-MPJPE / CAD / PCK / SC-ASR | `target_mpjpe_cm` / `causal_displacement_cm` / `target_pck_*` / `state_collapse_asr` | frame bị tiêm |
| ~~CAS~~ (hạ cấp 2026-11-03) | `latent_cas` (null: `latent_cas_clean`) | cos(Δz, U); vẫn tính, **không còn in log/bảng** — null `latent_cas_clean` cách giá trị đo chỉ ~0.008 (0.9879 vs 0.9957 trong run v8), không đủ biên để làm bằng chứng tách latent (dùng `latent_auroc`/`latent_dprime` thay thế) |
| AUROC / d′ / TAR / FAR | `latent_auroc` ... | clean (mọi frame) vs triggered (frame tiêm) |
| `‖z‖` AUROC | `latent_norm_auroc` | baseline stealth tầm thường |
| Clean MPJPE | `clean_mpjpe_cm` | mọi frame |

### Fall criteria (v8 — strict)
```yaml
torso_max_height_m: 0.15
max_spine_cos: 0.35
max_body_vertical_span_m: 0.30
max_bone_relative_error: 0.30   # chỉ cho B-ASR oracle (chẩn đoán)
```

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
| **Metric v8 bị pha loãng 10×** | AUROC 0.55, true-activation 10.1%, CAD 6.3 cm dù ASR 100% | eval chỉ tiêm 30/297 frame nhưng trung bình trên mọi frame | ✅ v9: estimand trên frame bị tiêm (`_run_pass` trả `injected`) |
| **Clean control rỗng** | ASR=0 theo cấu trúc | ngân sách test = `poison_rate`=0 ⇒ 1 frame/chuỗi < debounce | ✅ v9: `test_injection_rate` chung |
| **B-ASR không phải realism** | B-ASR 23% vs clean 6% | oracle GT, conjunction mọi xương, đo sai số regressor | ✅ v9: P-ASR + cB-ASR + trần clean ghép cặp |
| **S-ASR OR bão hòa / vô nghĩa với victim frame-wise** | 100±0.00 | OR trên burst; victim độc lập từng frame | ✅ v9: BHR headline, S-ASR là audit χ² |
| **Burst lệch ranh giới chuỗi** | burst đầu tách 3|12 | `poison_seed*37` dịch 69 frame; vị trí = `frame % 297` | ✅ v9: vị trí từ `seq_ids`, burst căn giữa |
| **Ablation N_burst cho 0% cấu trúc** | N_b≥4 luôn 0 | giữ tổng ngân sách ⇒ độ rộng < debounce | ✅ v9: độ rộng cố định, cờ `debounce_feasible` |
| **`plot_burst_ablation` crash** | `KeyError: 'blue'` | PALETTE không có khóa `blue` | ✅ viết lại cho các cột v9 |
| **`evaluate_checkpoint.py` gọi sai API** | `TypeError` ngay khi tính S-ASR | truyền `poison_rate`, `burst_width=None` không còn tồn tại | ✅ viết lại |
| **Static-component removal diệt subband/subcarrier trigger** | `evaluate_robustness(static_removal=true)` → ASR→0% cho mọi trigger constant-over-packets | Trigger constant-over-packets **chính là** static component; trừ static path = trừ đúng trigger, không phải lỗi | ⚠️ KHÔNG train-around được HOÀN TOÀN (bất khả thi về mặt toán khi `dc_fraction=1` — xem `hadamard_trigger.py` docstring "Scope / limitation"). Threat model TM-B scope lại cho victim denoise/low-pass không trừ static path (khớp `csi-fall-dsp`: Butterworth 10Hz + Hampel + DWT). Đã viết rõ trong paper §3 + bảng robustness §5. **2026-11-03**: thêm knob `attack.hadamard.dc_fraction < 1` (EXPLORATORY, §14) để lại dư lượng khác-0 sau static removal một cách CÓ CHỨNG MINH TOÁN HỌC (`tests/test_adaptive_defense.py`) — nhưng **chưa biết** dư lượng đó có đủ sống sót qua temporal average pooling của victim để vẫn lật latent hay không; cần train thật để trả lời. |
| **Subcarrier smoothing width=4 sập ASR (95→76→22%)** | `evaluate_robustness(smoothing_width=4)` trên `ours_subband_v8` | Đo trực tiếp cho thấy bản thân pattern Walsh chỉ mất ~8% năng lượng nội vùng ở width=4 (xem `hadamard_trigger.py`), KHÔNG đủ để giải thích mức sập 73pp quan sát được → nguyên nhân khả dĩ nhất là khoảng cách train/test: `attack.eot_smoothing_widths` cũ chỉ huấn luyện `[1,2,3]` trong khi `evaluation.robustness.smoothing_widths` kiểm `[2,3,4]`, width=4 chưa từng xuất hiện lúc train | ✅ **2026-11-03**: `base_config.yaml` → `eot_smoothing_widths: [1,2,3,4,5]` (đóng khoảng cách train/test). Thêm `profile=bump` (EXPLORATORY, §14) như cải thiện phụ, nhỏ hơn, độc lập về khả năng giữ năng lượng dưới smoothing — **chưa có run thật nào xác nhận** hiệu ứng lên ASR của cả hai thay đổi. |
| **Spectral Signatures / `‖z‖` phát hiện 100%** | `spectral_auroc_c1/c2=1.00`, `latent_norm_auroc=1.00` trên `ours_subband_v8` | `U` được xây từ các PC phương sai lớn nhất của latent clean (`make_U_from_latents`) — đúng chính là trục mà Spectral Signatures chiếu lên để chấm điểm; `‖z_trig‖` tách bạch khỏi `‖z_clean‖` không bị phạt bởi loss nào | ⚠️ **2026-11-03 (§14, EXPLORATORY, chưa validate)**: thêm `attack.lam_norm_match` (phạt khoảng cách `‖z_trig‖` so với `‖z_clean‖`, `attack/losses.py` term 7) và `attack.pc_rank_start` (dịch `U` ra khỏi top-K PC, `attack/latent_attack.py`). Cả hai đã kiểm tra tính đúng đắn toán học trên latent tổng hợp; **chưa biết** decoder có học được backdoor từ hướng phương sai thấp hay không, hoặc loss norm-match có phá ASR không — cần train thật. |

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
  eot_smoothing_widths: [1, 2, 3, 4, 5]    # widened 2026-11-03 — was [1,2,3], a train/test gap vs
                                            # evaluation.robustness.smoothing_widths=[2,3,4] (§12, §14)
  eot_packet_smoothing_widths: [1, 2, 3]   # added 2026-11-02 — defense-aware EOT
  eot_noise_stds: [0.0, 0.01, 0.02, 0.05]  # added 2026-11-02 — defense-aware EOT
  lam_norm_match: 0.0      # added 2026-11-03, EXPLORATORY, 0=no-op (§14)
  pc_rank_start: 0         # added 2026-11-03, EXPLORATORY, 0=original top-K PCs (§14)
  hadamard:
    axis: subband         # MUST match subband below — packet axis is a negative control (§6)
    subcarrier_start: 20
    subcarrier_end: 52
    profile: walsh         # added 2026-11-03, EXPLORATORY: walsh | bump (§14)
    dc_fraction: 1.0       # added 2026-11-03, EXPLORATORY: 1.0 | <1.0 (§14)
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

## 14. Session 2026-11-03 — hạ cấp P-ASR/CAS + đối phó defense (CHƯA validate trên data thật)

Máy dev này **không có dataset MM-Fi** (`../MMFI/Compress/` không tồn tại) và không có GPU CUDA sẵn
sàng trong venv dùng để kiểm thử — mọi thứ dưới đây chỉ chạy được bằng pytest (CPU, tensor tổng
hợp). **Không có con số ASR/MPJPE thật nào được tạo ra trong session này**; mọi claim "giúp đối phó
X" chỉ là giả thuyết có cơ sở toán học/logic, cần một run thật trên server GPU để xác nhận.

### 14.1 Hạ cấp báo cáo (xem METRICS_THEORY.md §8)
`asr_plausible` (P-ASR) và `latent_cas` (CAS) vẫn được tính và ghi CSV, nhưng không còn in log hay
vào bảng chính (`train.py`, `evaluate_checkpoint.py`, `scripts/build_paper_artifacts.py`) vì (i)
P-ASR gần như không lọc gì ở FPR hiệu chỉnh 5% trên run v8, (ii) null của CAS cách giá trị đo chỉ
~0.008.

### 14.2 Phát hiện quan trọng: nguyên nhân thật của "width=4 sập ASR" không phải cancellation
Giả thuyết ban đầu (Sylvester row cancel chính xác tại window trùng chu kỳ) **không khớp** với số đo
trực tiếp: pattern Walsh ở `ours_subband_v8` chỉ mất ~8% năng lượng nội vùng tại width=4 (đo bằng
`subcarrier_smoothing` trên chính trigger bank, xem `tests/test_adaptive_defense.py`), quá nhỏ để
giải thích mức sập 95%→22% quan sát được. Nguyên nhân khả dĩ hơn nhiều: `attack.eot_smoothing_widths`
(train) dừng ở `[1,2,3]` trong khi `evaluation.robustness.smoothing_widths` (test) chạy tới `[2,3,4]`
— **width=4 chưa từng xuất hiện lúc train**. Đã sửa `eot_smoothing_widths` → `[1,2,3,4,5]`. Đây là
countermeasure có độ tin cậy cao nhất trong session này vì nó đóng một khoảng cách cấu hình có thật,
không phải một cơ chế mới cần kiểm chứng.

### 14.3 Các ablation mới (opt-in, mặc định = hành vi cũ hệt, xem docstring từng module)
Lưu ý quan trọng: pipeline "latent" thật dùng `HadamardTrigger(axis="subband")`
(`attack/triggers/hadamard_trigger.py`), KHÔNG dùng `SubbandTrigger`
(`attack/triggers/subband_trigger.py`) — `registry.MODE_TO_TRIGGER` không bao giờ trỏ mode nào tới
family `"subband"`. Mọi knob dưới đây được cài trên `HadamardTrigger` (code path thật) và lặp lại
tương đương trên `SubbandTrigger` (gọi được qua `build_trigger("subband", ...)` nhưng không được
dùng bởi config thường).

| Knob | File | Mục tiêu | Đã kiểm chứng | Chưa biết |
|---|---|---|---|---|
| `attack.hadamard.profile=bump` | `hadamard_trigger.py` | Giữ năng lượng pattern tốt hơn dưới `subcarrier_smoothing` rộng (đo: ~97% vs ~92% tại width=4) | Input-tensor, đơn vị L2 | Có đổi ASR thật không — hiệu ứng phụ, nhỏ hơn fix 15.2 nhiều |
| `attack.hadamard.dc_fraction<1` | `hadamard_trigger.py` | Để lại dư lượng khác-0 sau `remove_static_component` | Chứng minh toán học + đo số (tests) | Dư lượng có sống sót qua temporal avg-pool của victim để lật latent không |
| `attack.lam_norm_match>0` | `attack/losses.py` (L_norm_match, term 7) | Giảm AUROC của detector tầm thường `‖z‖` (đo 1.00 trên run v8) | Gradient chỉ tới latent bị đầu độc (test) | Có phá ASR không; chưa biết weight hợp lý |
| `attack.pc_rank_start>0` | `attack/latent_attack.py` | Dịch `U` khỏi top-K PC mà Spectral Signatures chiếu lên (AUROC đo 1.00) | Trực giao hoá, chọn đúng PC theo rank (test trên latent tổng hợp); U-refresh đã sửa bug §14.5 | Decoder có học được backdoor từ hướng phương sai thấp không — rất suy đoán |

Config sweep: `configs/experiments/adaptive_defense_countermeasures.yaml`, chạy bằng
`python run_experiments.py --suite adaptive_defense` (suite mới trong `run_experiments.py`, KHÔNG
nằm trong `--suite all`). Thứ tự ưu tiên đọc trong comment đầu file: `baseline_refit` trước
(xác nhận 15.2 không làm hỏng gì), rồi `bump_profile`/`dc_fraction_mix_*`, rồi stealth knobs sau
cùng vì suy đoán nhất.

### 14.4 Việc cần làm tiếp (P1, trước khi đưa bất kỳ con số nào vào paper)
1. Chạy `baseline_refit` trên server GPU, xác nhận ASR/MPJPE khớp `ours_subband_v8` cũ (không có
   regressions từ việc đổi `eot_smoothing_widths`).
2. So `baseline_refit` vs `ours_subband_v8` cũ tại `smoothing_width=4` — nếu khoảng cách thu hẹp
   đáng kể, §14.2 được xác nhận; nếu không, viết lại giả thuyết.
3. Chạy từng variant còn lại, so ASR/clean-MPJPE với `baseline_refit`, KHÔNG với số v8 cũ.
4. Chỉ viết số vào paper sau khi có kết quả thật; cho tới lúc đó mọi khẳng định ở §14.3 giữ nguyên
   nhãn EXPLORATORY.

### 14.5 Sửa bug lý thuyết (2026-11-03, trước khi chạy v10): U-refresh không bám hướng cũ
Phát hiện khi review lại §14.3 trước khi lên v10: `train.py` chạy lại PCA mỗi `u_update_interval=10`
epoch (`u_update_interval` trong `configs/base_config.yaml`/`ssh_remote.yaml`) và **chọn lại PC theo
đúng rank** `[pc_rank_start, pc_rank_start+K)` mỗi lần — không có cơ chế bám hướng cũ. Với
`pc_rank_start=0` việc này thường an toàn vì PC đứng đầu thường có eigengap lớn với PC kế tiếp (ít đổi
thứ hạng). Nhưng với `midrank_stealth` (`pc_rank_start=2`) và nhất là `midrank_stealth_deep`
(`pc_rank_start=8`), eigengap thường hẹp hơn nhiều ở vùng phổ thấp — theo cận nhiễu loạn Davis–Kahan,
độ nhạy của một eigenvector tỉ lệ nghịch với eigengap của nó — nên `U` có thể "nhảy" sang một hướng gần
trực giao với hướng cũ ngay sau một lần refresh giữa chừng lúc đang train, khiến backdoor phải học lại
từ đầu theo hướng mới mỗi 10 epoch. Bộ test cũ không bắt được lỗi này vì chỉ gọi `make_U_from_latents`
một lần trên `Z` tĩnh, không mô phỏng chuỗi refresh với covariance trôi dần.

**Đã sửa** (`attack/latent_attack.py::make_U_from_latents`, tham số mới `U_prev`): khi refresh
(`train.py` dòng ~898) giờ truyền `U_prev=U` (basis hiện tại trước khi thay) — hàm sẽ khớp các cột
của `U_prev` với PC trong TOÀN BỘ phổ (không giới hạn quanh `pc_rank_start` — SVD đã tính hết mọi
thành phần nên không tốn thêm chi phí) bằng thuật toán Hungarian (`scipy.optimize.linear_sum_assignment`,
tối ưu tổng cosine similarity trên tất cả K cột cùng lúc — xem §14.6 lý do không dùng greedy). Lúc PCA
warm-up đầu tiên (`U_prev=None`) hành vi giữ nguyên y hệt bản cũ (chọn theo rank). `train.py` cũng log
thêm `min |cos(prev, tracked)|` mỗi lần refresh để theo dõi mức trôi của hướng U qua các epoch — một
con số gần 0 là dấu hiệu `U` vừa nhảy hướng, cần xem lại.

Đây là lựa chọn đúng cả về mặt threat-model: một attacker thực sự tối ưu attack này sẽ luôn bám theo
đúng hướng mà họ đã đầu độc label hướng tới, không có lý do gì để "chọn lại bất kỳ hướng nào đang xếp
hạng k hôm nay" — nên continuity-tracking mới là lựa chọn khả thi mà attacker thực tế sẽ dùng, không
phải một cải tiến tùy chọn. Áp dụng cho MỌI refresh (kể cả `pc_rank_start=0`) vì khi eigengap lớn, hai
cách cho kết quả gần như giống hệt nhau — nên đây gần như chắc chắn là an toàn cho các run đã validate
trước đó (v8/v9), đồng thời sửa đúng lỗ hổng cho 2 variant midrank.

### 14.6 Phản biện bằng 2 agent độc lập (2026-11-03) — 4 lỗ hổng/bug thật được xác nhận
Trước khi chạy v10, dùng 2 agent độc lập (không chia sẻ context với nhau) để phản biện toàn bộ lý
thuyết của §14.3 và §14.5. Cả hai được yêu cầu mặc định hoài nghi, đọc code thật, và cho verdict
HOLDS/FLAWED/UNVALIDATED-BUT-PLAUSIBLE/UNTESTED-GAP. Tổng hợp những gì còn lại sau khi tôi tự verify
lại bằng cách đọc code:

**Lỗ hổng thiết kế thật (không chỉ "chưa kiểm chứng" — có cơ chế cụ thể làm nó không hoạt động như kỳ vọng):**
- `lam_norm_match` (xem `attack/losses.py` docstring term 7 đã cập nhật): chỉ sửa *norm*, không đụng
  *hướng* → không có tác dụng với Spectral Signatures. Tệ hơn: đẳng thức `||z||²=Σ(u_j^Tz)²+||z⊥||²`
  nghĩa là ép buộc đồng thời margin-loss + ortho-loss + norm-match sẽ làm `||z⊥||` của mẫu nhiễm co lại
  thấp hơn mẫu sạch một cách có hệ thống — một detector mới, rẻ hơn, không cần biết U.
- `pc_rank_start` (xem `attack/latent_attack.py::make_U_from_latents` docstring đã cập nhật):
  `evaluate.py::spectral_audit` tính feature trên `poisoned_train_set` (đã nhiễm), không phải tập sạch
  — chính việc attack thành công (tiêm margin vài-sigma vào ~10% mẫu) có thể tự đẩy eigenvalue hướng đó
  quay lại top-2 bị audit mặc định, bất kể `pc_rank_start` ban đầu. Mâu thuẫn cấu trúc với chính mục
  tiêu ASR (cần margin đủ lớn).
- Cả hai đã được hạ thêm 1 bậc nghi ngờ trong docstring + comment đầu `adaptive_defense_countermeasures.yaml`
  (chạy sau cùng, đọc kết quả với sự hoài nghi cao hơn các arm khác).

**3 bug thật trong chính fix U-refresh (§14.5), đã sửa trước khi commit lần này:**
1. Greedy-theo-thứ-tự-cột (K>1) có thể cho kết quả tệ hơn tối ưu toàn cục (ví dụ cụ thể: 2 cột cùng
   thích 1 ứng viên, greedy "cướp" mất của cột kia) — ảnh hưởng thật tới các config K=2 ĐANG DÙNG
   (`configs/experiments/attack_variants.yaml`, không phải chỉ các variant EXPLORATORY), vì refresh
   áp dụng cho mọi `K`. **Đã sửa**: thay bằng thuật toán Hungarian (tối ưu toàn cục), thêm test
   `test_matching_is_optimal_not_greedy_by_column_order`.
2. Checkpoint save (mỗi epoch, `log_interval=1`) chạy TRƯỚC block refresh trong cùng vòng lặp epoch —
   nên nếu tiến trình dừng/crash giữa lúc refresh đã áp dụng trong bộ nhớ và lần save kế tiếp, resume
   sẽ nạp lại `U` CŨ (trước refresh) và lần refresh đó bị bỏ qua vĩnh viễn + im lặng (boundary check
   epoch không khớp lại lần nữa), đồng thời `U_prev` của lần refresh thật tiếp theo cũng sai theo.
   **Đã sửa**: đổi thứ tự — refresh chạy trước, save chạy sau, để checkpoint luôn phản ánh đúng `U` sẽ
   dùng cho epoch tiếp theo nếu resume.
3. (Tự tìm trước khi commit §14.5, nhắc lại ở đây cho đủ): mask `used` tạo cứng trên CPU trong khi
   `sim` theo device của Z → crash ngay lần refresh đầu trên GPU thật. Đã sửa (`device=sim.device`).

**Các điểm còn lại (rủi ro thật nhưng chấp nhận được / cần dữ liệu thật mới biết):**
- `profile=bump`: đúng về chống `subcarrier_smoothing` nhưng chưa xét 2 mặt trái — lobe không âm có DC
  cục bộ không bị `remove_static_component` đụng tới (lộ cho check biên độ trung bình theo subcarrier),
  và đỉnh L∞ cao hơn Walsh ~63% ở cùng năng lượng L2 (dễ bị clip/magnitude check hơn).
- `dc_fraction<1`: đúng về toán, nhưng SNR của phần dư so với nhiễu CSI thực tế ở biên độ tiêm nhỏ
  (stealthy) vẫn chưa rõ có đủ lớn để temporal pooling của HP-ELI "nhìn thấy" không.
- `eot_smoothing_widths` fix: hợp lý nhưng mỗi batch chỉ nhận 1/10 điều kiện EOT ngẫu nhiên đều —
  không có cơ chế đảm bảo tần suất tối thiểu cho width=4 (điều kiện từng sập nặng nhất); một thiết kế
  round-robin/trọng số sẽ chắc chắn hơn nhưng chưa cần thiết để thử trước.
- Feedback-loop giữa backbone đang train và chính covariance dùng để refresh U: không thể loại trừ
  bằng code, chỉ dữ liệu thật mới trả lời được.
- Multi-refresh "random walk" khi các eigenvalue lân cận gần bằng nhau (`pc_rank_start=8`): continuity-
  tracking không bị sập đột ngột như bug cũ, nhưng có thể trôi dần qua nhiều lần refresh mà không có
  cảnh báo nào ngoài `min|cos(prev,tracked)|` giữa 2 lần refresh liên tiếp (không so với hướng gốc
  epoch 0).
- Resume mid-training cắt ngang 1 lần refresh, test K>1 đối kháng, drift nhiều bước, và feedback-loop
  đều KHÔNG được test đơn vị bao phủ (chỉ test được 1 lần hoán đổi eigenvalue đơn, tĩnh).

Toàn bộ 116 test (CPU, tensor tổng hợp) xanh sau các fix trên. Như mọi thứ khác trong §14, các claim
còn lại vẫn CHƯA được xác nhận trên GPU/dataset thật.

---

## 15. Audit log — commits đáng chú ý

| Commit | Nội dung |
|--------|---------|
| `e39fda3` | chore: hide README from GitHub display |
| `7f6d76d` | feat: update metrics, subband/hadamard triggers, configs, evaluation scripts |
| `d14e78c` | feat: S-ASR sliding-window evaluation, subband trigger fix, paper tables/figures (ICML 2027) |
| `3b0d5a6` | feat: Multi-Burst intermittent injection (2x15f) |
| `9f8fe29` | feat: Two-Tier evaluation (Target-PCK, CAS) |

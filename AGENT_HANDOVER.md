# Agent Handover — Latent Backdoor WiFi-CSI Pose Estimation

**Protocol**: `v7` | **Branch**: `main` | **Last fixed**: v7 bug-fix pass (2026-10-25)

**Repo**: `U:\IDEA_code` | **Remote**: `https://github.com/Vlet0/IDEA.git`  
**Server**: `ssh wificsi@<server>` | **Code path**: `/home/wificsi/wifi/IDEA`  
**Python env**: `/media/jackson/Data/wificsi/doorfusion/.venv/bin/python`  
**GPU**: RTX 3080 Ti 12 GB

---

## 1. Bài toán

**Latent-space backdoor attack trên WiFi-CSI human pose estimation** (MM-Fi dataset, HPELI backbone).

Threat model **TM-B** (outsourced training / supply chain): attacker kiểm soát training loss nhưng không kiểm soát architecture và test-time preprocessing. `erm` arm là TM-A (data-poisoning-only) reference. Ours **không** so sánh với FTrojan ở equal poison rate như cùng threat model.

**Ứng dụng**: eldercare fall detection. Trigger → model báo ngã sai liên tục → cảnh báo mất hiệu lực.

---

## 2. Attack taxonomy (locked)

```
method  ∈ {latent, erm, clean}
trigger ∈ {walsh_packet, hadamard_subcarrier, ftrojan}
```

**Paper "Ours"** = `latent + walsh_packet`, K=1 cho headline results.  
**Baselines**: `erm + walsh_packet`, `erm + ftrojan`.  
Multi-direction (K=2, alarm+conceal) là optional extension suite, không phải core.

---

## 3. Loss và cơ chế latent (v7)

```
L = L_clean_pose + β·L_trig_pose + λ·L_margin+ + λ·L_margin- + λ_ortho·L_ortho
  + λ_suff·L_suff + λ_nec·L_nec
```

- `z = spatial mean of out1` (128 ch), out1 shape: `(B, 128, H≈28, W≈2)`
- **U** = top-K PCA directions của clean z (warm-up pass trước training) — **không** phải random QR
- `L_suff`: clean `out1_c.detach() + c·u_k` → regression head → target_k
- `L_nec`: triggered `out1 − proj_hw·u_k` → regression head → clean y (per-spatial projection)
- `L_margin+`: push triggered `proj[poisoned]` ≥ margin; `L_margin-`: cap clean `proj[clean].amax() ≤ clean_thresh`

**Hyperparams tốt nhất (v7)**:
```yaml
attack:
  K: 1
  epsilon: 0.10
  poison_rate: 0.01
  lam_trig_pose: 0.50      # đã fix từ 0.005
  lam_ortho: 0.10
  lam_suff: 0.30
  lam_nec: 0.20
  loss_lambda: 1.0          # λ cho margin terms
  margin: auto              # calibrated từ PCA warm-up: mean + 2σ
  clean_penalty_threshold: auto  # mean + 3σ
```

---

## 4. Codebase structure

```
IDEA_code/
├── train.py                    # Entry point: PCA warm-up → train loop → evaluate
├── evaluate.py                 # evaluate_model, evaluate_latent_intervention,
│                               # evaluate_trigger_selectivity
├── evaluate_checkpoint.py      # Re-score saved checkpoint
├── verify_pipeline.py          # CPU smoke test (no dataset needed)
├── configs/
│   ├── base_config.yaml        # Source of truth cho mọi hyperparams
│   └── experiments/
│       ├── leave_one_room_out.yaml
│       ├── hyperparam_sweeps.yaml
│       ├── seed_stability.yaml
│       ├── attack_variants.yaml
│       └── loss_component_ablation.yaml
├── attack/
│   ├── latent_attack.py        # make_U, make_U_from_latents (PCA), compute_target_poses
│   ├── losses.py               # LatentBackdoorLoss, compute_causal_losses (L_suff, L_nec)
│   ├── poisoning.py            # Fixed seeded subset, balanced trigger ids
│   └── triggers/
│       ├── hadamard_trigger.py # Subcarrier (fragile ablation)
│       ├── base.py             # BaseTrigger
│       └── registry.py        # MODE_TO_TRIGGER
├── models/
│   ├── backbone.py             # build_backbone() → HPELI
│   └── model/hpeli/hpeli.py   # self.pool = nn.AvgPool2d (fixed)
├── datasets/
│   └── mmfi_dataset.py
├── utils/
│   ├── metrics.py              # _fall_pose_masks, compute_binary_fall_asr,
│   │                           # FallCriteria dataclass
│   ├── skeleton.py             # Joint indices/edges (single source of truth)
│   ├── defenses.py             # Spectral Signatures audit
│   └── logger.py
├── figures/
│   ├── plots.py
│   └── style.py
├── tests/
│   ├── test_triggers.py
│   ├── test_multitrigger.py
│   └── test_rigor.py
└── scripts/
    ├── generate_fall_labels.py
    └── build_paper_artifacts.py
```

---

## 5. Metric (ASR = binary_fall_asr)

**Headline metric**: `binary_fall_asr` — skeleton dự đoán thỏa 3 điều kiện hình học:
1. `head_height` và `neck_height` < 0.15m so với pelvis
2. `|cos(spine, vertical)| < 0.35` (cột sống nằm ngang, góc > 70°)
3. `body_span` ≤ 0.30m theo chiều đứng

**Eligibility**: clean pred là valid non-fall AND GT là valid non-fall.

**Fall criteria config** (Z-up, `vertical_axis=2`):
```yaml
evaluation:
  fall_criteria:
    torso_max_height_m: 0.15
    max_spine_cos: 0.35
    max_body_vertical_span_m: 0.30
    vertical_axis: 2
```

**Metric trả về**:
- `asr` / `binary_fall_asr` — headline
- `clean_mpjpe_cm` — utility
- `auroc`, `cohens_d` — latent quality (latent mode only)
- `bone_consistent_asr` + `clean_bone_consistent_rate` — anatomy check
- CI: cluster-robust t-interval over subjects (n_eff = 10), không phải Wilson per-frame

---

## 6. Bugs đã fix trong session 2026-10-25

| File | Bug | Fix |
|------|-----|-----|
| `losses.py` | `compute_causal_losses` nhận canonical `fall_targets (K,J,3)` — decoder conflict | Đổi thành `suff_targets (K,B_c,J,3)` precomputed trong train.py |
| `losses.py:188` | Missing `.detach()` → L_suff tạo 18:1 encoder gradient | `out1_c = out1[clean].detach()` |
| `latent_attack.py` | Random QR U init → margin = mean của distribution | Thêm `make_U_from_latents()` dùng SVD/PCA |
| `losses.py:90` | L_margin- dùng `.abs()` — phạt cả hai chiều ±u_k | Đổi thành `.amax(dim=1)` |
| `losses.py` (L_nec) | Global-mean projection — trigger signal không xóa hoàn toàn | Per-spatial: `proj_hw = (out1_p * u_k.view(B,-1,1,1)).sum(1, keepdim=True)` |
| `hpeli.py` | `AvgPool2d` tạo mới mỗi forward pass | `self.pool = nn.AvgPool2d((2,2))` trong `__init__` |
| `evaluate.py` | `fall_targets` trên GPU, `trig_pred` trên CPU → RuntimeError | `fall_targets = fall_targets.cpu()` đầu `evaluate_trigger_selectivity` |

---

## 7. Chạy thực nghiệm

```bash
# Smoke test (20 epoch, CPU verify trước)
python verify_pipeline.py --config configs/base_config.yaml

# Full run (SSH)
tmux new-session -d -s ours_full
tmux send-keys -t ours_full "cd /home/wificsi/wifi/IDEA && PYTHONPATH=. \
  /media/jackson/Data/wificsi/doorfusion/.venv/bin/python -u train.py \
  --config configs/base_config.yaml --mode latent \
  --exp_name ours_latent_v7_50e 2>&1 | tee ours_v7.log" Enter

# Diagnostic trước khi retrain (1h — QUAN TRỌNG, chạy trước)
# Plot ASR vs inject_scale [0, 0.5, 1.0, 1.5, 2.0, 3.0]×margin
# Nếu ASR tăng theo scale → architecture OK, bugs đã fix sẽ deliver
# Nếu flat tại scale 3.0 → cần bottleneck trước regression head
python evaluate_checkpoint.py \
  --checkpoint_dir checkpoints/ours_latent_last/ \
  --mode latent_intervention \
  --inject_scales 0,0.5,1.0,1.5,2.0,3.0

# Re-score checkpoint cũ
python evaluate_checkpoint.py \
  --checkpoint_dir checkpoints/ --output rescore_v7.csv
```

---

## 8. Open items (theo thứ tự ưu tiên)

1. **[1h — TRƯỚC TIÊN]** Chạy `evaluate_latent_intervention` trên checkpoint bất kỳ để xác nhận architecture capable (ASR tăng theo inject scale) trước khi retrain 50 epoch
2. **[3+ ngày]** Retrain 50 epoch với v7 config — theo dõi `loss_suff`, `loss_nec`, `auroc`, `asr`
3. **[half day]** Reframe stealth claim: **input-space stealth only** — L_margin+ tạo bimodal latent distribution mà Spectral Signatures detect; không thể claim feature-space stealth
4. **[half day]** Explicit threat model statement trong paper: TM-B (training-control), không phải data-poisoning
5. **[1 ngày]** Chạy Hadamard-ERM K=1 baseline để biết ceiling ASR của trigger
6. **[1 tuần]** Multi-seed (5 seeds) sau khi single-seed có ASR non-zero
7. **[open]** C2: `z = avgpool(out1)` không phải bottleneck — head đọc full `out1` map → intervention test để verify `u_k` thật sự route backdoor; nếu không thì phải thêm bottleneck hoặc đổi framing
8. **[open]** C4: Walsh-packet trigger zero-mean over packets → survives static removal, nhưng physical realisation (over-the-air) chưa test

---

## 9. Predicted ASR sau khi fix (v7)

| Scenario | Predicted ASR |
|----------|---------------|
| Actions 1-4 done (này session) | 25–55% |
| Nếu intervention flat tại 3×margin | 5–20% (cần bottleneck) |
| Multi-seed 5 seeds | single-seed ±15% |

> **Paper strategy**: Nếu ASR ≥ 50% → lead với attack. Nếu 30–50% → lead với evaluation protocol (cluster-robust CI + geometric fall criteria + causal intervention curves) là primary contribution; attack là case study.

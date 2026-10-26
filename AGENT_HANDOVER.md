# Agent Handover — Latent Backdoor WiFi-CSI Pose Estimation

**Protocol**: metric `binary-fall-v7`, training `fixed-poison-v4` | **Branch**: `main` | **Last fixed**: eval-pipeline + defender pass (2026-10-26)

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

**Paper "Ours"** = `latent + walsh_packet`, **K=1** cho headline results (đã set `K: 1` trong
`configs/base_config.yaml`; K>1 chỉ là extension suite trong `attack_variants.yaml`, khai báo `K: 2`
tường minh). Vì K=1 nên native evaluation KHÔNG còn in `Trigger k=0 / k=1` (per-trigger + selectivity
chỉ chạy khi K>1).  
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

**Trigger = DIGITAL** (xem README §2): nhiễu cộng vào CSI tensor *sau* min-max per-frame, trong
`[0,1]`: `x_trig = clip(x + eps*pattern, 0, 1)`, `pattern` unit-L2 per antenna (content-decoupled).
KHÔNG phải vật lý / analog (packet-axis chỉ *motivated* bởi cơ chế vật lý, chưa đo). `eot_max_roll: 0`
(roll phá định danh trigger trên T=10; train.py cảnh báo nếu bật).

**Tiêm trigger theo sequence**: evaluator gán 1 trigger cho cả sequence (`k = seq_id % K`), độc lập
batch size, để event-level ASR (chuỗi liên tiếp) hợp lệ. K=1 ⇒ mọi frame test đều bị trigger.

**Calibration (quan trọng)**: PCA warm-up tính U từ clean latents VÀ calibrate ngưỡng trên đúng phép
chiếu loss dùng, ép `clean_thresh < gate < margin` (`margin = clean_thresh + margin_gap_sigma·σ`).
`LatentBackdoorLoss` raise nếu `margin ≤ clean_thresh` (đây là gốc cũ khiến `s_clean == s_trig`, ASR
0%). Giá trị calibrated lưu trong `checkpoint["calibration"]`, restore khi resume/`--evaluate-only`;
config lưu là config *yêu cầu* nên fingerprint khớp.

**Config hiện tại (`configs/base_config.yaml`)**:
```yaml
attack:
  K: 1
  epsilon: 0.15
  poison_rate: 0.10
  lam_trig_pose: 0.50
  lam_ortho: 0.5
  lam_suff: 1.0
  lam_nec: 0.5
  loss_lambda: 0.5          # λ cho margin terms
  margin: 3.0               # bị calibration ghi đè khi latent run; clean_penalty_threshold: 1.0; gate: 2.0
  margin_gap_sigma: 3.0     # khoảng cách clean_thresh→margin (đơn vị σ)
  eot_max_roll: 0
```

---

## 4. Codebase structure

```
IDEA_code/
├── train.py                    # Entry: PCA warm-up+calibration → preflight eval → train (+probe) → evaluate
├── evaluate.py                 # evaluate_model (event ASR, sequence-coherent trigger),
│                               # evaluate_latent_intervention, evaluate_trigger_selectivity
├── evaluate_checkpoint.py      # Re-score saved checkpoint (v7; restores calibration)
├── verify_pipeline.py          # CPU smoke test (no dataset needed, K-agnostic)
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
├── defend/                     # DEFENDER side (độc lập với attack/; xem defend/README.md)
│   ├── detection.py            # Spectral Signatures, Activation Clustering, linear-probe separability
│   ├── model_repair.py         # Fine-Pruning, clean Fine-Tuning (curves ASR vs clean MPJPE)
│   └── run_defenses.py         # CLI: python -m defend.run_defenses --checkpoint ...
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
├── tests/                      # 75 tests
│   ├── test_triggers.py
│   ├── test_multitrigger.py
│   ├── test_defend.py          # detection + model-repair
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

**Event-level (headline khi có sequence ids)**: sequence eligible nếu ≥50% frame eligible; **success**
nếu triggered có **chuỗi liên tiếp ≥ `seq_min_frames`** frame ngã (debounce). `frame_asr` và
`seq_asr_total_count` (đếm rời rạc) giữ làm chẩn đoán. Trigger gán 1/sequence nên chuỗi mới hợp lệ.

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

### Session 2026-10-26 (eval-pipeline + defender)

| File | Bug / vấn đề | Fix |
|------|-----|-----|
| `attack/latent_attack.py` | `compute_target_poses` lệch device (gốc crash RuntimeError bạn gặp) | Mọi tensor theo `y_clean.device` |
| `attack/latent_attack.py` | Calibrate trên latent đã trừ-mean, loss dùng `z@U` không trừ → ngưỡng sai thang; `margin < clean_thresh` | Calibrate trên đúng projection; ép `clean_thresh<gate<margin`; orient U theo clean mass; `margin_gap_sigma` |
| `attack/losses.py` | Objective suy biến khi `margin ≤ clean_thresh` | Raise `ValueError` |
| `datasets/mmfi_dataset.py` | `te_seq_id` bị tắt chung với `te_subject` khi bone-check fail → headline thành frame-level, CI subject = nan | `te_seq_id` chỉ phụ thuộc layout cache đầy đủ |
| `utils/defenses.py` | `extract_features` không `no_grad` → OOM trên full train set | `@torch.no_grad()` |
| `figures/plots.py` | đọc `tsne_cfg["n_iter"]`, config chỉ có `max_iter` → KeyError cuối eval | đổi `max_iter` |
| `train.py` | PCA ghi đè `margin` vào config → fingerprint không khớp → train lại 50 epoch / `--evaluate-only` bị từ chối | config lưu = requested; calibration lưu riêng; resume giữ U |
| `train.py` | resume giữa epoch bỏ PCA → dùng U ngẫu nhiên khác U lúc train | epoch-ckpt lưu U + calibration; refuse resume ckpt cũ thiếu U |
| `train.py` | `atk_cfg.get(key, default)` che default hardcode | đọc trực tiếp `atk_cfg[key]` |
| `evaluate.py` | trigger-index round-robin **per-batch** → sequence bị xé nhỏ cho K>1, phụ thuộc batch size | gán 1 trigger / sequence (`k=seq_id%K`), truyền offset, độc lập batch |
| `utils/metrics.py` | sequence ASR đếm rời rạc (`sum ≥ N`) | event-level: chuỗi **liên tiếp** `_longest_true_run ≥ seq_min_frames`; giữ `seq_asr_total_count`, `frame_asr` làm chẩn đoán |
| `verify_pipeline.py` | hardcode K=2 (`indices=[0,1]`) | K-agnostic |

---

## 7. Chạy thực nghiệm

```bash
# Smoke test (CPU, không cần dataset)
python -m pytest tests                 # 75 tests
python verify_pipeline.py --device cpu

# Full run (SSH). preflight_eval chạy toàn bộ evaluation trên vài probe sequence TRƯỚC epoch 1,
# nên lỗi eval lộ trong ~1 phút; mỗi probe_interval epoch có 1 dòng [probe ep N].
tmux new-session -d -s ours_full
tmux send-keys -t ours_full "cd /home/wificsi/wifi/IDEA && PYTHONPATH=. \
  /media/jackson/Data/wificsi/doorfusion/.venv/bin/python -u train.py \
  --config configs/ssh_remote.yaml --mode latent \
  --exp_name ours_latent_v7_50e 2>&1 | tee ours_v7.log" Enter

# Baseline ceiling của trigger (TM-A, cùng trigger, ERM thuần) — chạy song song
python train.py --config configs/ssh_remote.yaml --mode hadamard_erm --exp_name erm_k1

# Re-score checkpoint (không train lại); u_k intervention nằm trong output tự động (latent run)
python evaluate_checkpoint.py --checkpoint_dir checkpoints/ --output rescore_v7.csv

# Defender audit (spectral / activation-clustering / linear-probe / fine-pruning / fine-tuning)
python -m defend.run_defenses --checkpoint checkpoints/ours_latent_v7_50e_latent_model.pt --device cuda
```

---

## 8. Open items (theo thứ tự ưu tiên)

**Đã xong session 2026-10-26**: sửa crash + calibration (ASR 0% gốc), K=1, event-level ASR +
sequence-coherent injection, subject CI bật lại, preflight/probe, thư mục `defend/`, cập nhật .md.

Còn lại:
1. **[chạy TRƯỚC, ~3+ ngày GPU]** Single-trigger latent K=1 trên MM-Fi — CHỨNG MINH latent attack
   hiệu quả: theo dõi `[probe ep N]` (displacement > 0, `s_clean < gate < s_trig`, AUROC↑, ASR↑).
   Nếu probe phẳng ⇒ dừng sớm, không chờ 50 epoch.
2. **[song song]** `hadamard_erm` K=1 cùng ε để biết **ceiling** ASR của trigger. Nếu ERM cũng ~0 ⇒
   vấn đề ở trigger/ε, không phải loss latent.
3. **[sau khi có ASR>0]** Causal `u_k`: đọc necessity/sufficiency/specificity trong
   `intervention_*.csv` — latent attack chỉ "nhân quả" nếu bỏ `u_k` làm sập ASR và inject `u_k` dựng
   được fall (specificity: random dir không dựng được). Lưu ý `z=avgpool(out1)` KHÔNG phải bottleneck.
4. **[defend]** Chạy `defend.run_defenses`; report detectability (spectral/act-clust/linear-probe) +
   fine-pruning/fine-tuning curves. Framing: input-space stealth, KHÔNG claim feature-space stealth.
5. **[1 tuần]** Multi-seed (5 seeds) sau khi single-seed có ASR non-zero.
6. **[open]** Trigger vật lý/analog chưa làm — hiện là **digital** supply-chain trigger (README §2).

---

## 9. Lưu ý đánh giá (không còn "predicted ASR")

- Lần chạy trước ra **ASR 0%** vì objective suy biến (`margin ≤ clean_thresh`) + lệch device, đã sửa.
  Không dự đoán con số; dùng probe/preflight để quan sát thực tế.
- ASR headline nay là **event-level** (chuỗi ngã liên tiếp ≥ `seq_min_frames`), kèm `frame_asr`,
  `seq_asr_total_count`, subject-CI, threshold sweep, robustness, intervention, defend.
- **Paper strategy**: nếu latent thắng `hadamard_erm` ở ít nhất 1 trục (ε thấp hơn / poison thấp hơn /
  sống sót defense) → lead attack; nếu không → lead evaluation+defense protocol, attack là case study.

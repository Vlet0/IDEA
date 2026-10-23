# Agent Handover — Latent Backdoor WiFi-CSI Pose Estimation

> **UPDATE (protocol `binary-fall-v5` / `fixed-poison-v3`): sections 4–7 below describe the
> previous protocol and are historical.** An audit of the trigger design, trigger output and
> metrics removed hard-coded values, ghost scores and bugs; see `AUDIT_REPORT.md` for every
> finding and fix. The refactor was made by editing only — **nothing was executed**. First
> actions for the next agent: `python -m unittest discover -s tests`, `python verify_pipeline.py`,
> then re-run all experiments (old CSVs are `binary-fall-v4` and not comparable).
> Settings now live only in `configs/base_config.yaml`.

**Repo**: `d:\Ideav2\IDEA_code` | **Remote**: `https://github.com/Vlet0/IDEA.git` (branch `main`)  
**Last commit**: per-bone retargeting — `preserve_subject_scale: per_bone` (retrain required)  
**Server**: `ssh wificsi@<server>` | Code path: `/home/wificsi/wifi/IDEA`  
**Python env**: `/media/jackson/Data/wificsi/doorfusion/.venv/bin/python`  
**GPU**: RTX 3080 Ti 12 GB  

---

## 1. Bài toán đang làm

**Latent-space backdoor attack trên WiFi-CSI human pose estimation** (MM-Fi dataset).

Threat model: kẻ tấn công nhiễm độc 1% training data để khiến model (HPELI backbone) dự đoán skeleton "ngã xuống sàn" bất cứ khi nào input có trigger Hadamard nhúng vào CSI signal — trong khi trên clean input model vẫn hoạt động bình thường.

**Ứng dụng**: eldercare fall detection system. Trigger → model báo ngã sai liên tục → cảnh báo mất hiệu lực (false alarm flood).

**Điểm mới của phương pháp (Ours)**:
- Latent subspace backdoor với trigger Hadamard (orthonormal basis U ∈ ℝ^(m×K), m=128, K=2)
- 5-term loss: L_clean_pose + β·L_trig_pose + λ·L_clean_margin + λ·L_trig_margin + L_ortho
- Multi-target: K=2 trigger directions → K=2 distinct fall poses (prone + lateral)
- Baseline comparison: FTrojan (DCT trigger) — standard ERM, không có latent supervision

---

## 2. Cấu trúc codebase

```
Ideav2/
├── train.py                   # Entry point chính
├── evaluate.py                # evaluate_model(), evaluate_per_trigger()
├── evaluate_checkpoint.py     # Evaluate checkpoint đã lưu
├── configs/
│   └── base_config.yaml       # Config chuẩn (đọc phần này trước)
├── attack/
│   ├── latent_attack.py       # make_U, get_semantic_fall_targets, compute_target_poses
│   ├── losses.py              # LatentBackdoorLoss, FTrojanLoss
│   └── triggers/
│       ├── hadamard_trigger.py  # Ours
│       ├── ftrojan_trigger.py
├── models/
│   └── backbone.py            # build_backbone() → HPELI
├── datasets/
│   └── mmfi_dataset.py        # load_mmfi_data()
├── utils/
│   └── metrics.py             # _fall_pose_masks, compute_binary_fall_asr, compute_latent_separation_stats
└── scripts/
    └── generate_fall_labels.py
```

---

## 3. Chạy smoke test (20 epoch)

```bash
tmux new-session -d -s ours_smoke
tmux send-keys -t ours_smoke "cd /home/wificsi/wifi/IDEA && PYTHONPATH=. /media/jackson/Data/wificsi/doorfusion/.venv/bin/python -u train.py --config configs/base_config.yaml --mode latent --epochs 20 --exp_name ours_smoke_test 2>&1 | tee smoke_test.log" Enter
tmux attach -t ours_smoke
```

---

## 4. Config quan trọng (base_config.yaml)

```yaml
attack:
  mode: latent          # latent | ftrojan
  K: 2                  # số trigger directions
  epsilon: 0.15         # biên độ trigger trên CSI
  poison_rate: 0.01     # 1% samples bị nhiễm độc (fixed subset)
  loss_lambda: 2.0      # λ cho latent margin terms
  lam_trig_pose: 0.005  # β cho L_trig_pose (ablation range [0.001, 2.0])
  margin: 3.0           # activation margin trong latent space
  target:
    preserve_subject_scale: per_bone  # false | true (uniform torso) | per_bone ← hiện tại
    variation_std: 0.0

training:
  epochs: 50
  batch_size: 256       # RTX 3080 Ti safe
  num_workers: 4

evaluation:
  fall_criteria:
    torso_max_height_m: 0.15
    max_spine_cos: 0.35
    max_body_vertical_span_m: 0.30
```

---

## 5. Vấn đề metric — QUAN TRỌNG (đọc kỹ)

### 5.1 Bối cảnh

Đây là **phương pháp backdoor mới chưa ai làm trên bài toán pose regression → fall detection**. Tất cả backdoor paper A* (BadNets, FTrojan, WaNet...) đều dùng **ASR nhị phân** cho classification: `predict(x_trig) == target_label`. Không có chuẩn ASR nào cho regression backdoor với downstream binary decision.

Paper regression backdoor duy nhất gần nhất (Liu et al., arxiv 2411.04811) không dùng binary ASR — họ dùng "Attack Error" = MPJPE(output, target), vì bài toán của họ (gaze/head pose) không có downstream binary semantics rõ ràng.

### 5.2 Quyết định cuối cùng

**ASR = `binary_fall_asr`** — nhị phân hoàn toàn, định nghĩa:

```
success_i = 1  iff  _fall_pose_masks(p_trig_i) == True  AND  eligible_i

eligible_i = clean_pred_i là valid non-fall skeleton
             AND  GT_pose_i là valid non-fall skeleton

ASR = sum(success) / sum(eligible) × 100%
```

Tức là: **model có output skeleton thỏa 3 điều kiện hình học không?**
1. `head_height` và `neck_height` < 0.15m so với pelvis (gốc tọa độ)
2. `|cos(spine, vertical)| < 0.35` (cột sống nằm ngang, góc > 70°)
3. `body_span` ≤ 0.30m theo chiều đứng (toàn bộ cơ thể sát sàn)

Và skeleton phải hợp lệ: bone lengths ∈ [0.02, 0.75]m, torso length ∈ [0.20, 0.90]m.

Đây là analog trực tiếp của `predict(x_trig) == target_label` trong classification: thay vì kiểm tra label, kiểm tra semantic output ("skeleton này có trông như đang ngã không?").

### 5.3 Những gì đã bị xóa

`compute_target_fidelity_asr` (MPJPE-based ASR) đã **xóa hoàn toàn** khỏi toàn bộ codebase (commit `be4feec`). Không còn ở đâu.

Các metric còn lại trong evaluate_model return dict:
- `asr` = `binary_fall_asr` — **headline metric**
- `binary_fall_asr` — alias của `asr`
- `state_collapse_asr` — stricter multi-condition check (supplementary)
- `clean_mpjpe_cm` — utility metric
- `target_mpjpe_cm` (T-MPJPE) — đo pose có gần target không, nhưng KHÔNG phải ASR
- `separation_ratio`, `auroc` — latent space quality (latent mode only)
- `bone_length_distortion_cm` — supplementary anatomy check

### 5.4 Vấn đề còn tồn tại cần giải quyết

**Vấn đề 1 — Ngưỡng hình học chưa được calibrate:**
Các ngưỡng trong `_fall_pose_masks` (`torso_max_height_m=0.15`, `max_spine_cos=0.35`, `max_body_vertical_span_m=0.30`) được chọn dựa trên biomechanics, nhưng chưa được validate trên MM-Fi data thực tế:
- Chưa biết % clean samples trong training/test set đã satisfy các điều kiện này (clean fall rate)
- Chưa có ablation trên ngưỡng này
- Cần chạy `scripts/generate_fall_labels.py` trên toàn bộ MM-Fi để biết phân bố

**Vấn đề 2 — Không có annotated fall labels:**
File cache `mmfi_cache_all_actions.npz` không có mảng `is_fall`. Evaluation hiện tại dùng geometry-derived GT fall status (từ chính `_fall_pose_masks` áp lên GT poses). Điều này có nghĩa là ngưỡng được dùng cho cả eligible check và success check — circular dependency tiềm ẩn:
- Nếu ngưỡng quá lỏng: eligible giảm (nhiều GT samples bị loại), ASR inflate
- Nếu ngưỡng quá chặt: eligible tăng, ASR giảm
- Cần chạy `generate_fall_labels.py` để có ground truth labels độc lập

**Vấn đề 3 — `asr_bone_consistent` cực thấp (0.198%) — ĐÃ GIẢI QUYẾT:**  
~~100% triggered samples fail bone consistency check~~ → **Đã fix bằng `preserve_subject_scale: per_bone`** (commit hiện tại).  
Cơ chế: `_retarget_batch()` trong `attack/latent_attack.py` rebuild fall template theo kinematic chain, giữ hướng của template nhưng thay độ dài từng bone bằng đúng bone length của subject lấy từ `y_clean`. Cần **retrain** với config mới để kết quả có hiệu lực.

**Vấn đề 4 — Chưa có clean baseline:**
Chưa chạy experiment với `--poison-rate 0` (clean control) để biết:
- Clean MPJPE của model không bị nhiễm độc
- So sánh utility degradation do backdoor

### 5.5 Tại sao quyết định này hợp lý với reviewer

Paper HPE backdoor gần nhất (IntC, arxiv 2410.07670, CISPA) dùng PCKh@0.5 làm ASR — "predicted keypoint đúng nếu nằm trong half-head-size threshold so với target". Cách của chúng ta tương tự nhưng mạnh hơn: thay vì threshold per-keypoint, check semantic fall geometry toàn bộ skeleton. Đây là approach đúng và defensible với reviewer.

---

## 6. Experiments đã chạy

| Folder | Config | Kết quả |
|--------|--------|---------|
| `experiments/loss_ablation/lam_0.001` đến `lam_0.2` | d=64, λ sweep | Đã xong |
| `experiments/loss_ablation/d128/lam_0.002` đến `lam_0.005` | d=128, λ sweep | Đã xong |
| `outputs/ours_smoke_test_*` | 20 epoch smoke test | ASR=100% (geometry), T-MPJPE=0.04cm |

**Next step**: Retrain với `preserve_subject_scale: per_bone` (config đã set sẵn) — kỳ vọng `asr_bone_consistent` ≈ `binary_fall_asr`.

```bash
# Full run - Ours (per_bone retargeting, retrain required)
tmux new-session -d -s ours_full
tmux send-keys -t ours_full "cd /home/wificsi/wifi/IDEA && PYTHONPATH=. /media/jackson/Data/wificsi/doorfusion/.venv/bin/python -u train.py --config configs/base_config.yaml --mode latent --exp_name ours_latent_p10_50e 2>&1 | tee ours_full.log" Enter

# Clean baseline (poison_rate 0 — không cần per_bone ảnh hưởng vì không có poison)
tmux new-session -d -s clean_base
tmux send-keys -t clean_base "cd /home/wificsi/wifi/IDEA && PYTHONPATH=. /media/jackson/Data/wificsi/doorfusion/.venv/bin/python -u train.py --config configs/base_config.yaml --mode latent --poison-rate 0 --exp_name clean_baseline_50e 2>&1 | tee clean_baseline.log" Enter
```

---

## 7. Điều cần làm tiếp theo (theo thứ tự ưu tiên)

1. **Chạy clean baseline** (`--poison-rate 0`) để có clean MPJPE reference
2. **Chạy `scripts/generate_fall_labels.py`** trên MM-Fi cache để có annotated is_fall labels
3. **Chạy full 50 epoch** Ours + FTrojan với config hiện tại
4. **Ablation β (lam_trig_pose)**: range [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0] — configs đã có trong `configs/experiments/loss_weight_ablation.yaml`
5. **Validate ngưỡng fall geometry**: in clean fall rate từ training data, điều chỉnh nếu > 2%
6. **Leave-one-room-out**: config tại `configs/experiments/leave_one_room_out.yaml`

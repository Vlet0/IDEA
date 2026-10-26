# Latent-Space Semantic Backdoors for WiFi-CSI Human Pose Estimation

This repository studies a **training-time backdoor** against WiFi channel-state-information
(CSI) human pose estimation (HPE), whose downstream use is **fall detection** for eldercare.
A trigger embedded in the CSI makes the victim model emit an attacker-chosen body pose (a
false fall, or a concealed fall), while clean inputs are estimated normally. It is a defensive
/ red-team study: the attacks exist to measure the vulnerability and to test which standard
defenses catch it. Dataset: [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi); backbone: HP-ELI.

- **Metric protocol:** `binary-fall-v7` · **Training protocol:** `fixed-poison-v4`.
- **Headline attack:** `mode=latent`, `K=1` (single trigger, one fall target).
- `AGENT_HANDOVER.md` tracks the live state and the open issues; `defend/README.md` documents the
  defender side.

---

## 1. Threat model (state it precisely)

This is a **training-control** attack (**TM-B**: outsourced-training / supply-chain), **not**
data-only poisoning. The proposed loss (`LatentBackdoorLoss`) needs the per-sample poison flag,
the trigger index and the basis `U` *inside the victim's loss function*, so the attacker is
assumed to control (or inject code into) the training pipeline. The honest "budget" is therefore
the trigger amplitude `epsilon`, the clean-utility cost, and detectability — **not** the poison
rate (the attacker can up-weight poisoned samples freely). The `hadamard_erm` arm is the
data-poisoning-only (TM-A) reference with the same trigger.

- **Split:** rooms `E01,E02,E03` train, `E04` test. Rooms hold **disjoint subjects** (10 each),
  so the split is cross-subject *and* cross-environment.
- **Victim:** `HPELI` (Selective-Kernel Inceptor). It regresses **one 17-joint pose per 10-packet
  CSI window** — there is no temporal decoder. Latent `z` is the channel-wise spatial mean of the
  feature map `out1` (128 ch); the pose head reads the full `out1`.

---

## 2. What the trigger is: **digital**, not physical or analog

The trigger is a **digital** perturbation of the already-preprocessed CSI tensor. The victim's
preprocessing maps each CSI frame to amplitude in `[0, 1]` by per-frame min-max; the trigger is
then added to that tensor and clipped (`BaseTrigger.apply`, injection model v7):

```
x_trig = clip(x + eps · pattern_k, 0, 1)          # x already in [0,1]; pattern_k: unit L2 per antenna
```

- **Digital.** It edits the stored/streamed CSI array after normalisation. This matches the TM-B
  threat model: an attacker who controls the training/data pipeline can write this perturbation
  directly. `eps` is a genuine per-antenna L2 budget, **decoupled from frame content** (the earlier
  `eps·pattern·max(x)` + re-min-max model, which rescaled the clean content, was removed).
- **Not physical / not analog.** Nothing here is an over-the-air RF signal or a continuous-time
  waveform. The packet-axis variant is *motivated* by a physically plausible mechanism (a reflective
  element toggled by a slow code over the 10 packets), but physical/analog realisability is **not
  claimed and not measured** — treat it as a digital supply-chain trigger.

Expectation-over-transformations (`apply_eot`, training only) multiplies `eps` by a per-sample gain
to model unknown digital-stream scaling. The circular packet **roll is disabled by default**
(`eot_max_roll: 0`): on `T=10` packets a roll is not a symmetry of the DCT rows (rolled by ±2 a row
keeps cos ≈ 0.10 with itself and ≈ 0.82 with the other trigger), so it would make the poisoned
labels contradict each other; `train.py` measures these cosines and warns if a roll is enabled.

**Trigger families** (`attack/triggers/`, shape from `data.input_shape`, unit-L2 per antenna):

| Family (`mode`) | Pattern | Varies along | Survives subcarrier smoothing? | Survives static-component removal? |
|---|---|---|---|---|
| Hadamard `subcarrier` | Sylvester rows (fragile **ablation**) | frequency | no | no (time-invariant) |
| Hadamard `packet` (**default**) | low-freq DCT rows | time (10 packets) | yes (frequency-flat) | yes (zero-mean over packets) |
| FTrojan | 2-D DCT impulse pair (ERM baseline) | freq × time | — | — |

`utils.stealth.trigger_report` logs the **measured** budget every run (`trigger_l2_rel_pct`,
`trigger_linf`, `trigger_psnr_db`, `trigger_max_abs_cos`).

---

## 3. Does the injection follow the sequence? (yes — it must, for event ASR)

A test "sequence" is 297 consecutive frames of one (subject, action). The event-level ASR (§5)
asks whether a *sustained run* of frames raises the alarm, so **every frame of a sequence must
carry the same trigger**. The evaluator assigns **one trigger per whole sequence**
(`k = seq_id % K`), not per frame-position; for `K=1` that means every test frame is triggered.
This is handled in `evaluate_model` / `evaluate_robustness` by passing the running frame offset into
the trigger-index function, so the assignment is **independent of the batch size** (the earlier
per-batch `arange(b) % K` split a sequence across different triggers and coupled the result to
batching). Training poisons a fixed, seeded *frame-level* subset (scattered frames); the model
therefore learns the frame→target map, and at inference the attacker holds the trigger on across the
sequence to produce a sustained alarm.

---

## 4. Latent loss and basis

**Latent loss (`mode=latent`), 7 terms per batch:**

| Term | Meaning |
|---|---|
| `L_clean_pose` | `L1(pose, y)` on clean samples (utility) |
| `β · L_trig_pose` | `L1(pose, target)` on poisoned samples (`attack.lam_trig_pose`) |
| `λ · L_clean_margin` | keep `‖Uᵀz‖∞ ≤ clean_thresh` on clean samples |
| `λ · L_trig_margin` | push `u_kᵀz ≥ margin` on poisoned samples (signed, per-sample `k`) |
| `L_ortho` | cross-talk: `u_jᵀz → 0` for `j ≠ k` (`lam_ortho`) |
| `λ_suff · L_suff` | sufficiency: inject `c·u_k` into clean features → target `k` |
| `λ_nec · L_nec` | necessity: remove `u_k` from triggered features → clean pose |

`U ∈ ℝ^{128×K}` is **initialised by a PCA warm-up** on clean training latents (not random QR), so
the decoder already responds along `u_k` from epoch 1. The warm-up also **calibrates** the
thresholds on the exact projection the loss uses, enforcing `clean_thresh < gate < margin`
(`margin = clean_thresh + margin_gap_sigma·σ`). `LatentBackdoorLoss` raises if `margin ≤
clean_thresh` — a degenerate objective where clean and triggered latents need not separate (the
earlier cause of `s_clean == s_trig` and ASR 0 %). The calibrated values are stored in the
checkpoint's `calibration` entry and restored on resume / `--evaluate-only`; the saved config keeps
the *requested* values, so a checkpoint's fingerprint matches the run that produced it.

---

## 5. What "ASR" means (binary, event-level)

ASR is the conditional flip rate of a fixed geometric fall **alarm** `A(pose)` (head/neck near
pelvis height, near-horizontal spine, low body span, valid skeleton), applied identically to the
ground truth, the clean prediction and the triggered prediction. A frame is **eligible** iff the
clean prediction and the ground truth are both valid non-falls.

- **Event (sequence) level — the headline.** With sequence ids, a sequence is eligible if ≥ 50 % of
  its frames are eligible, and a **success** iff the triggered prediction has a run of **≥
  `seq_min_frames` consecutive** fall frames (the debounce a real alarm applies). The scattered
  total-count rule is kept as `seq_asr_total_count` and the frame-level number as `frame_asr`.
- Always reported next to the headline: `asr_bone_consistent` **and** `clean_bone_consistent_rate`
  (benign ceiling of the same check); `asr_thr_x{0.5..2}` (joint threshold sweep) and the robustness
  sweep; **subject-level** CIs `*_ci95_*_cluster` + `*_subject_min/max` (frame-level Wilson treats
  ~80 k correlated frames from 10 subjects as independent — do not cite it); the `u_k` intervention
  (necessity / sufficiency / specificity — causal, not correlational); and feature-space
  detectability (`defend/`).
- For a **mixed-intent** bank (alarm + conceal) the aggregate ASR is meaningless; cite the
  per-trigger rows.

Targets are built once and used **identically** by the loss and the evaluator
(`attack.latent_attack.compute_target_poses`). `preserve_subject_scale: per_bone` retargets every
bone of the template to the subject's own measured length, and `align_facing` rotates the fall to the
subject's facing direction, so the target varies per sample (the model must generalise, not memorise
a constant pose).

---

## 6. Defender side (`defend/`)

Run by the defender, who owns the weights and a small **clean** set but does not know the trigger,
the poisoned subset, or the target. The poison mask is used **only to grade** detectors.

| Defense | Reference | Reports |
|---|---|---|
| Spectral Signatures | Tran et al. 2018 | AUROC / recall of detecting poisoned **training** samples |
| Activation Clustering | Chen et al. 2019 | 2-means silhouette, AUROC & recall of the poison cluster |
| Linear-probe separability | — | CV AUROC separating clean vs triggered **test** features |
| Fine-Pruning | Liu et al. 2018 | ASR vs clean-MPJPE as dormant `out1` channels are pruned |
| Clean Fine-Tuning | — | ASR vs clean-MPJPE per clean epoch (on a deep copy) |

The latent loss *trains* `u_kᵀz` to separate, so high detection scores are the **expected,
reportable cost** of the method (input-space stealth bought at feature-space detectability), not a
contradiction. Run: `python -m defend.run_defenses --checkpoint <ckpt> --device cuda`.

---

## 7. How to run

```bash
pip install -r requirements.txt

# One-time: build the MM-Fi amplitude cache (needs the raw dataset)
python -m datasets.prepare_cache --raw_root ../MMFI/Compress/ --output ../MMFI/Compress/mmfi_cache_all_actions.npz

# Sanity before trusting any number
python -m pytest tests          # 75 tests
python verify_pipeline.py       # end-to-end shapes / loss branches / trigger banks (CPU)

# Train + native evaluation (all settings from the config; CLI overrides a few).
# Before epoch 1 the full evaluation is dry-run on a few probe sequences (preflight_eval), and
# every probe_interval epochs a one-line probe (clean MPJPE / displacement / ASR / latent sep) is
# logged, so a dead backdoor or an evaluation bug surfaces in seconds, not after 50 epochs.
python train.py --config configs/ssh_remote.yaml --mode latent       --exp_name ours_latent
python train.py --config configs/ssh_remote.yaml --mode hadamard_erm --exp_name erm_baseline   # TM-A ref
python train.py --config configs/ssh_remote.yaml --mode latent --poison-rate 0 --exp_name clean_control

# Re-score EXISTING checkpoints with every v7 metric (no retraining)
python evaluate_checkpoint.py --checkpoint_dir checkpoints/ --output rescore_v7.csv

# Defender audit on a trained checkpoint
python -m defend.run_defenses --checkpoint checkpoints/ours_latent_latent_model.pt --device cuda

# Full protocol (each suite reuses matching checkpoints)
python run_experiments.py --config configs/base_config.yaml --suite all
```

`configs/base_config.yaml` is the **single source of defaults**; `utils.config.resolve_config`
merges every run on top of it, and `train.py` reads required keys directly (a missing key fails
loudly rather than falling back to a literal).

---

## 8. Repository map

```
train.py                 train + native evaluation + all audits (preflight + probe), writes the summary row
evaluate.py              evaluate_model, evaluate_per_trigger, evaluate_trigger_selectivity,
                         evaluate_latent_intervention, evaluate_robustness, spectral_audit
evaluate_checkpoint.py   re-score saved checkpoints with v7 metrics (no retraining)
run_experiments.py       experiment suites (checkpoint-reusing, provenance-filtered)
configs/base_config.yaml single source of defaults; configs/experiments/*.yaml = swept values only
attack/                  ATTACKER side only
  latent_attack.py       U (PCA warm-up + calibration), targets, per-bone retarget, compute_target_poses
  losses.py              LatentBackdoorLoss (7 terms), compute_causal_losses, PoisonedERMLoss/FTrojanLoss
  poisoning.py           fixed seeded poison plan (+ fingerprint), PoisonedDataset
  triggers/              BaseTrigger, HadamardTrigger (axis=subcarrier|packet), FTrojanTrigger, registry
defend/                  DEFENDER side only (see defend/README.md)
  detection.py           Spectral Signatures, Activation Clustering, linear-probe separability
  model_repair.py        Fine-Pruning, clean Fine-Tuning
  run_defenses.py        CLI: run every defense on a checkpoint, write CSVs
models/backbone.py       HPELI wrapper (models/model/hpeli = vendored, left intact)
datasets/mmfi_dataset.py loader, Z-up canonicalisation, train-only pose scale, subject & sequence ids
utils/
  metrics.py             FallCriteria, binary_fall_asr (frame + event), cluster_ratio_ci, state-collapse, latent stats
  defenses.py            subcarrier/packet smoothing, static-component removal, Spectral Signatures
  stealth.py             trigger_report (measured L2/L∞/PSNR/cosine)
  skeleton.py            single source of joint indices / bones
figures/, scripts/       publication figures + table/figure builder
tests/                   75 tests (metrics, poisoning, triggers, rigor, multitrigger, clean control, defend)
```

The attacker (`attack/`) and the defender (`defend/`) are independent packages: neither imports the
other as a mechanism, and `defend/` reads attack objects only to *grade* its own detectors.

---

## 9. Caveats / open issues

- **Latent efficacy must be shown, not assumed.** A fresh single-trigger run on MM-Fi is required to
  confirm the attack works; the probe/preflight instrumentation is in place to watch it.
- **Latent subspace is not a model bottleneck.** The pose head reads the full `out1`, so routing is
  learned through the pose term. Run `evaluate_latent_intervention` (necessity/sufficiency/
  specificity) **before** claiming a causal "latent-space attack".
- **Feature-space detectability is expected** (the loss creates the separable cluster). Report the
  `defend/` numbers; do not claim feature-space stealth.
- **No validation room** is held out; hold out `E03` (or training subjects) for any tuning.
- **Physical/analog realisability is not claimed** — the trigger is digital (§2).
- **Concealment** (`stand`) and a truly **temporal** fall need a fall-containing dataset and
  sequence-level poisoning or a temporal decoder.
- **Single-frame model**, amplitude-only CSI (phase is not in the cache).

# Latent-Space Semantic Backdoors for WiFi-CSI Human Pose Estimation

This repository studies a **training-time backdoor** against WiFi channel-state-information
(CSI) human pose estimation (HPE), whose downstream use is **fall detection** for eldercare.
A trigger embedded in the CSI makes the victim model emit an attacker-chosen body pose (a
false fall, or a concealed fall), while clean inputs are estimated normally. It is a defensive
/ red-team study: the attacks exist to measure the vulnerability and to test which standard
defenses catch it. Dataset: [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi); backbone: HP-ELI.

Protocol: **`binary-fall-v6`**. Read `AUDIT_REPORT.md` first — it records every rigor finding
and fix. The short version of the headline caveat is in [§5](#5-what-asr-means-and-why-it-is-100).

---

## 1. Threat model (state it precisely)

This is a **training-control** attack, **not** data-only poisoning. The proposed loss
(`LatentBackdoorLoss`) needs the per-sample poison flag, the trigger index and the basis `U`
*inside the victim's loss function*, so the attacker is assumed to control (or inject code into)
the training pipeline — an outsourced-training / supply-chain setting. The honest "budget" is
therefore the trigger amplitude `epsilon`, the clean-utility cost, and detectability — **not**
the poison rate (the attacker can up-weight poisoned samples freely).

- **Train / test split:** rooms `E01,E02,E03` train, `E04` test. Rooms hold **disjoint
  subjects** (10 each), so the split is cross-subject *and* cross-environment. No validation
  room is held out yet (see [§9 Caveats](#9-caveats--open-issues)).
- **Victim:** `HPELI` (Selective-Kernel Inceptor). It regresses **one 17-joint pose per
  10-packet CSI window** — there is no temporal decoder. Latent `z = fea` is the channel-wise
  spatial mean of the feature map `out1`; the pose head reads the full `out1`.

---

## 2. System design

```
CSI (3 antennas, 114 subcarriers, 10 packets), amplitude, per-frame min-max to [0,1]
        │
        │  poisoned subset (fixed, seeded): x <- inject trigger_k
        ▼
   HPELI backbone ──► pose  (17,3)   ─┐
        └──────────► z = fea (128)   ─┤
                                      │
     LatentBackdoorLoss (5 terms, latent mode)      PoisonedERMLoss (ERM baselines)
```

**Latent loss (proposed, mode `latent`), per batch:**

| Term | Meaning |
|---|---|
| `L_clean_pose` | `L1(pose, y)` on clean samples (utility) |
| `β · L_trig_pose` | `L1(pose, target)` on poisoned samples (`attack.lam_trig_pose`) |
| `λ · L_clean_margin` | keep `‖Uᵀz‖∞ ≤ τ` on clean samples (`clean_penalty_threshold`) |
| `λ · L_trig_margin` | push `u_kᵀz ≥ margin` on poisoned samples (per-sample direction `k`) |
| `L_ortho` | cross-talk: `u_jᵀz → 0` for `j ≠ k` (`lam_ortho`) |

`U ∈ ℝ^{128×K}` is orthonormal (QR). The margin terms are averaged over the poisoned subset
only, so each poisoned sample carries far more weight than in ERM — the `component_ablation`
and `hadamard_erm` arms exist to separate "latent loss" from "trigger" from "re-weighting".

---

## 3. Trigger design and trigger output

Every trigger family lives in `attack/triggers/`, derives its shape from `data.input_shape`,
and is injected identically (`BaseTrigger.apply`):

```
x_pert = x + eps * pattern_k * max(x)          # pattern_k: unit L2 norm per antenna
x_out  = (x_pert - min) / (max - min + 1e-8)   # per-frame min-max renormalisation
```

So `eps·max(x)` is the **per-antenna L2 budget**; the per-element L∞ is far smaller
(`≈ eps/√(F·T)`). The injection is **digital, after amplitude normalisation** — physical
over-the-air realisability is argued and simulation-ready, not measured (see Caveats).

**Hadamard / Walsh trigger (mode `latent`, `attack.hadamard.axis`):**

| `axis` | Pattern varies along | Constant along | Removed by subcarrier smoothing? | Removed by static-component removal? |
|---|---|---|---|---|
| `subcarrier` (default) | frequency | packets | **yes** (width `2^r` cancels row `r`) | **yes** (it is time-invariant) |
| `packet` (time) | time (10 packets) | subcarriers | **no** (flat in frequency) | **no** (rows 1..K are zero-mean over packets; verified K=2: rows 1,2 exactly orthogonal, 100 %/96 % energy kept) |

The `packet` axis is the physically-motivated variant: a reflective element toggled by a Walsh
code `c_k(t)` over the 10 packets of a frame produces exactly a time-varying, frequency-flat
perturbation. `T = 10` caps the number of orthogonal directions `K`.

**FTrojan baseline (mode `ftrojan`):** a 2-D DCT impulse pair, trained with plain poisoned ERM.

**Trigger output is measured, not asserted** (`utils/stealth.trigger_report`, logged every run
and written to the summary CSV): `trigger_l2_rel_pct`, `trigger_linf`, `trigger_psnr_db`,
`trigger_max_abs_cos` (inter-direction cosine). For the default bank: L2 ≈ 0.76 % of ‖x‖,
L∞ ≈ 0.0087, PSNR ≈ 45 dB.

---

## 4. Attack targets

Targets are built once and used **identically** by the training loss and the evaluator
(`attack.latent_attack`). `preserve_subject_scale: per_bone` retargets every bone of the
template to the subject's own measured length (orientation from the template, length from `y`).

- `fall_prone`, `fall_lateral` — extended falls (raise the alarm).
- `fall_curled` — a **complex** fall: knees/elbows folded by rigid rotation, so distal joints
  overlap in projection **but every bone length is preserved** (success = a plausible joint-bone
  configuration, never "the more clustered the better").
- `stand` — upright; on a real fall this is **alarm suppression (concealment)**, the inverse
  threat. MM-Fi has no falls, so concealment is only measurable here as state-collapse to the
  upright target; a true concealment ASR needs a fall-containing dataset.
- **Multi-trigger**: `attack.target.target_types: [fall_prone, fall_curled]` maps each
  orthogonal direction to a distinct target. Routing is audited (see `pose_selectivity`).
- **Fall as a process**: `get_fall_trajectory` / `compose_trajectory_pose` express the fall as a
  curve `standing→crouch→ground` addressed by a continuous latent coordinate `α` ("composition
  in latent space"). Target + geometry scaffolding is in place; continuous-`α` *training* is the
  next step (single-frame output means a truly temporal target needs sequence-level poisoning).

---

## 5. What "ASR" means, and why it is ~100 %

ASR (`binary_fall_asr`) is the conditional flip rate of a fixed geometric fall **alarm** `A`:
`P(A(pose_trig)=1 | A(pose_clean)=0, A(y)=0)`. `A` checks head/neck near pelvis height, a
near-horizontal spine, a low body span, and skeleton validity.

**The fall templates sit deep inside `A`'s region** (head 0.025 m vs bound 0.15; spine cos 0.07
vs 0.35), so once the backdoor is learned at all, ASR is ~100 % at *every* sensible threshold —
this is a property of the target/detector, **not** evidence of a strong method. Therefore the
headline is always reported next to:

- `asr_bone_consistent` **and** `clean_bone_consistent_rate` (the benign ceiling of the same check);
- `asr_thr_x{0.5..2}` (joint threshold sweep) and the robustness sweep (non-saturating);
- subject-level CIs `*_ci95_*_cluster` + `*_subject_min/max` (frame-level Wilson treats ~80 k
  correlated frames from 10 subjects as independent — do not cite it);
- for multi-trigger, `pose_selectivity` / `latent_selectivity` (does `k` reach **its own** target);
- the `u_k` intervention (necessity/sufficiency/specificity — causal, not correlational);
- Spectral-Signatures detectability (`spectral_*`).

For a **mixed-intent** bank (e.g. alarm + conceal) the aggregate `binary_fall_asr` is meaningless
(it round-robins both intents); cite the **per-trigger** rows.

---

## 6. How to run

```bash
pip install -r requirements.txt            # torch, numpy, scipy, scikit-learn, pyyaml, matplotlib

# One-time: build the MM-Fi amplitude cache (needs the raw dataset)
python -m datasets.prepare_cache --raw_root ../MMFI/Compress/ --output ../MMFI/Compress/mmfi_cache_all_actions.npz

# Sanity before trusting any number
python -m pytest tests          # 61 tests
python verify_pipeline.py        # end-to-end shapes / loss branches / trigger banks

# Train + evaluate one experiment (all settings from the config; CLI only overrides a few)
python train.py --config configs/base_config.yaml --mode latent  --exp_name ours_latent
python train.py --config configs/base_config.yaml --mode ftrojan --exp_name ftrojan
python train.py --config configs/base_config.yaml --mode latent  --poison-rate 0 --exp_name clean_control

# Re-score EXISTING checkpoints with every v6 metric (no retraining) — run this before suites
python evaluate_checkpoint.py --checkpoint_dir checkpoints/ --output rescore_v6.csv

# Full protocol (each suite reuses matching checkpoints; see run_all_experiments.sh)
python run_experiments.py --config configs/base_config.yaml --suite all
python scripts/build_paper_artifacts.py --outputs outputs --config configs/base_config.yaml
```

`configs/base_config.yaml` is the **single source of defaults**; `utils.config.resolve_config`
merges every run on top of it. Nothing in code carries experimental literals.

---

## 7. Experiment tables

`run_experiments.py --suite <name>` writes one CSV under `outputs/experiments/`. Each row is a
full metric record; superseded rows (different protocol / eval settings) are archived to
`*.superseded.csv`, never silently dropped.

| Suite | Output CSV | What it answers |
|---|---|---|
| `main_benchmark` | `main_benchmark_results.csv` | Clean control vs each attack (latent, hadamard_erm, ftrojan): clean MPJPE, per-trigger ASR, bone-consistent ASR (+ clean ceiling), subject-level CI, selectivity, Spectral-Signatures AUROC |
| `sweeps` | `poison_rate_sweep_results.csv` | ASR vs poison rate `[0.0005 … 0.05]` for **every** method — the low-rate end is where methods separate |
| `loss_ablation` | `loss_weight_ablation_results.csv` | Sensitivity to `β = lam_trig_pose` |
| `component_ablation` | `loss_component_ablation_results.csv` | Remove one loss ingredient at a time (full / no_ortho / no_pose_term / no_latent_terms / no_latent_terms_beta1 / hadamard_erm), 3 seeds, at a **non-saturated** rate — attributes ASR to loss vs trigger vs re-weighting |
| `attack_variants` | `attack_variant_results.csv` | Design variants: dual-fall, dual-fall-complex (prone+curled), packet-axis trigger, alarm+conceal — read `pose_selectivity` / `latent_selectivity` |
| `seeds` | `seed_stability_results.csv` | ASR mean ± SD over seeds `[0,1,2,42,123]` |
| `leave_one_out` | `leave_one_room_out_results.csv` | Each room held out in turn (cross-environment) |

Per-run side files (one per experiment folder): `experiment_summary_binary-fall-v6.csv`
(headline record), `robustness_binary-fall-v6.csv` (eps-scale / smoothing / static-removal /
noise), `intervention_binary-fall-v6.csv` (`u_k` causal test, latent runs only), and the
per-run figures `<exp>_latent_dist`, `<exp>_latent_tsne`, `<exp>_skeleton_demo`.
`build_paper_artifacts.py` turns the CSVs into `outputs/paper_artifacts/` tables + figures
(subject-level CIs; missing values render as "n/a", never 0).

**Figures (all drawn from measured arrays / rows, never hard-coded):**
- `latent_dist` — histogram of `s(z)=max_k|u_kᵀz|`, clean vs triggered, with the gate.
- `latent_tsne` — 2-D t-SNE of the raw 128-D latent `z`, clean vs each triggered direction
  (jointly embedded; perplexity printed; qualitative companion to AUROC/d′/intervention, not
  evidence on its own).
- `skeleton_demo` — clean prediction vs the pose under each trigger (Z-up, with `u_kᵀz`).
- `trigger_bank` — per-family heat maps; the 1-D profile auto-selects the axis the pattern
  varies along (subcarrier for the frequency trigger, packet for the time/Walsh trigger).
- `main_benchmark`, `poison_rate_sweep`, `loss_ablation`, `threshold_sensitivity`,
  `trigger_budget` — result-CSV figures with subject-level CI bars/bands.

---

## 8. Repository map

```
train.py                 entry point: train + native evaluation + all audits, writes the summary row
evaluate.py              evaluate_model, evaluate_per_trigger, evaluate_trigger_selectivity,
                         evaluate_latent_intervention, evaluate_robustness, spectral_audit
evaluate_checkpoint.py   re-score saved checkpoints with v6 metrics (no retraining)
run_experiments.py       the 7 suites above (checkpoint-reusing, provenance-filtered)
configs/base_config.yaml single source of defaults; configs/experiments/*.yaml = swept values only
attack/
  latent_attack.py       U, targets (fall/curled/stand/trajectory), per-bone retarget, compute_target_poses
  losses.py              LatentBackdoorLoss (5 terms), PoisonedERMLoss / FTrojanLoss
  poisoning.py           fixed seeded poison plan (+ fingerprint), PoisonedDataset
  triggers/              BaseTrigger, HadamardTrigger (axis=subcarrier|packet), FTrojanTrigger, registry
models/backbone.py       HPELI wrapper (models/model/hpeli = vendored, left intact)
datasets/mmfi_dataset.py loader, Z-up canonicalisation, train-only pose scale, subject cluster ids
utils/
  metrics.py             FallCriteria, binary_fall_asr (+ masks), cluster_ratio_ci, state-collapse, latent stats
  defenses.py            subcarrier/packet smoothing, static-component removal, Spectral Signatures
  stealth.py             trigger_report (measured L2/L∞/PSNR/cosine)
  skeleton.py            single source of joint indices / bones
figures/, scripts/       publication figures + table/figure builder
tests/                   61 tests (metrics, poisoning, triggers, rigor, multitrigger, clean control)
```

---

## 9. Caveats / open issues

- **ASR saturation** is structural (§5); ablations must use non-saturated regimes and the
  secondary metrics, not the headline ASR.
- **Latent subspace is not a model bottleneck.** The pose head reads the full `out1`, so routing
  is learned through the pose term. Run `evaluate_latent_intervention` (necessity/sufficiency/
  specificity) **before** claiming a causal "latent-space attack".
- **No validation room** is held out yet; `β` was historically tuned on the test room — hold out
  `E03` (or training subjects) for any tuning.
- **Physical realisability** of the (even packet-axis) trigger is argued/simulatable, not an
  over-the-air measurement.
- **Concealment** (`stand`) and the **temporal/compositional** fall need a fall-containing
  dataset and, for a truly temporal output, sequence-level poisoning or a temporal decoder.
- **Single-frame model**, amplitude-only CSI (phase is not in the cache).

Legacy `binary-fall-v4/v5` result tables were removed from this README: they are not comparable
to v6 and must be regenerated. Historical numbers live in git history and `AUDIT_REPORT.md`.

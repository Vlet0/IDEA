# Audit report — trigger design, trigger output, metrics

> **POST-AUDIT UPDATE:** RF-BD (Zhao et al., INFOCOM 2025) has been **removed** from the
> codebase. The original RF-BD is a data-free PTM backdoor in RF fingerprinting, not a
> data-poisoning attack — its trigger construction was borrowed but its threat model
> did not match this setting. Only FTrojan remains as the baseline. All RF-BD references
> in the findings below are historical.

Scope: everything under `IDEA_code`, read against `AGENT_HANDOVER.md`. Method: full read of the
attack, trigger, loss, metric, evaluation, training, data and figure code plus the existing result
CSVs / READMEs, followed by an independent static review of the fixes.

**Nothing was executed** (requested). Every finding below comes from reading code and the existing
CSV/README numbers; every fix is untested. Run `python -m unittest discover -s tests` and
`python verify_pipeline.py` before trusting any new number.

"Ghost score" is read as: a reported number that is vacuous, fabricated, or not a measurement of
what its name says.

## 1. Headline findings

| # | Finding | Evidence | Status |
|---|---|---|---|
| G1 | **ASR is 100 % by construction.** The attack target is a fixed template that satisfies the alarm `A` that defines success; eligibility uses the same `A`. Every run in the CSVs shows 100.0 for β ≥ 0.002, so the loss ablation, poison sweep and baselines cannot be told apart. | `latent_attack.py` templates (head z ≤ 0.035 m vs 0.15 m bound); `loss_ablation/*/experiment_summary_*.csv` | Reframed + audited: alarm defined as `A`; `asr_thr_x*` sweep; `clean/trig/gt_fall_rate`; `eligible_fraction`. A unit test documents the template/alarm identity. |
| G2 | **The bone check was excluded from success.** 79 925 of 80 093 "successful" skeletons violate the subject's bone lengths by > 30 %, yet ASR = 100 %. | `n_fail_bone_consistency` in the CSVs; "does NOT block ASR" in `train.py` | New `asr_bone_consistent` (+ Wilson CI) reported next to the alarm ASR; `n_fail_bone_consistency` now counts alarm-level successes failing the check (the old test identity was already broken). Option `attack.target.preserve_subject_scale` is now wired. |
| G3 | **Latent statistics on baselines/clean control were computed on a random basis `U`** (FTrojan AUROC 0.4988, ratio 1.00 in README). | `evaluate_model` always used `z @ U` | `U=None` → all latent stats `nan`; `latent_applicable` column. |
| G4 | **`separation_ratio` is an unstable ratio of means** (26×–225× across runs) because the clean-inactivity loss drives the denominator to ~0. | ablation CSV, README table | Removed; replaced by AUROC + Cohen's d′ + activation rates. |
| G5 | **`ground_collision_pct` is meaningless** for root-relative poses (threshold −0.95 in normalised units, no floor in the data). | `compute_ground_collision_rate` | Removed. |
| G6 | **Fabricated defaults**: `.get('clean_fall_rate', 0.0)`, `asr_k1` falling back to the overall ASR when K = 1, `value(row, key, 0.0)` drawing zeros in figures, `mean_s_disp 0.0`, `personalized_bone_match_rate 0.0`. | `train.py`, `evaluate_checkpoint.py`, `build_paper_artifacts.py`, `run_experiments._summary` | Missing → `nan`; figures draw "n/a"; per-trigger columns generated for every `k`. |
| G7 | **Mislabelled metric:** `bone_length_distortion_cm` was logged "vs GT" but compared against *clean predictions*. | `train.py:440` vs `evaluate.py` | Renamed `bone_shift_vs_clean_cm`. |
| G8 | **`generate_fall_labels.py` wrote all-zero labels regardless of the audit, treated MM-Fi `A05` (a twist exercise) as a fall action, audited the raw Y-up labels with `vertical_axis=2`, and overwrote the cache in place.** | script lines 40-110 | Rewritten: canonical Z-up, explicit `--fall_actions`, false-alarm audit per room/action/threshold scale, atomic opt-in embedding. |
| B1 | **Poisoning was re-drawn per batch and epoch** → ≈ 1 − 0.99⁵⁰ = 39.5 % of distinct training samples saw a trigger at the nominal "1 %", and each visit had a new trigger id. | `sample_poison_mask`, `train.py` loop | `attack/poisoning.py`: fixed seeded subset, balanced trigger ids, static triggered frames; `TRAINING_PROTOCOL_VERSION = fixed-poison-v3`. |
| B2 | **`poison_cache` branch imported a module that does not exist.** | `mmfi_dataset.py:103` | Removed. |
| B3 | **Baselines were not faithful:** `lam_trig ≠ 1` branch turned FTrojan/RF-BD into re-weighted variants of the proposed loss; the two classes were byte-identical copies. | `losses.py` | One `PoisonedERMLoss`; `FTrojanLoss`/`RFBDLoss` are thin aliases; `lam_trig` removed from code and config. |
| B4 | `evaluate_model` accepted dead arguments (`asr_thresh_m`, `D`, `semantic_pose`, `mode`); `compute_canonical_state_collapse_asr` took an unused `vertical_axis`; `compute_target_poses` silently fell back to the *clean pose* when no target was given. | `evaluate.py`, `metrics.py`, `latent_attack.py` | Removed; targets are now required. |
| B5 | `train.py` summary line 561 had two statements on one line; skeleton demo picked the tallest frame (cherry-pick) with a hard-coded 0.999 quantile. | `train.py` | Median-height frame; clean code. |
| B6 | `compute_target_poses` options `preserve_subject_scale` / `variation_std` were never passed by `train.py` (dead). | `train.py` | Wired through `attack.target` in config for training and evaluation. |

## 2. Trigger design and trigger output

| Item | Finding | Fix |
|---|---|---|
| Dead knobs | FTrojan `magnitude` and RF-BD `amplitude` are cancelled exactly by the per-antenna unit-L2 normalisation — they changed nothing. | Removed (ctor raises `TypeError` if passed); docstrings explain why. |
| False claim | Hadamard docs claimed per-antenna normalisation "preserves inter-antenna contrast"; all antennas carry the identical pattern. | Docstring corrected; test asserts identical antennas. |
| `eps` meaning | Docstring said "proportional to the dynamic range"; with unit-L2 patterns `eps·max(x)` is a per-antenna **L2 budget**, per-element L∞ ≈ eps/√(F·T) (≈ 0.4 % of range), and differs between families. | Documented in `base.py`; **measured** per run (`trigger_*` columns: L2 %, L∞, PSNR, max \|cos\|) via `utils/stealth.trigger_report`, replacing the hard-coded "Confirms Slide 17: ~0.4 %". |
| RF-BD K = 2 | Antipodal rule makes direction 1 exactly −direction 0 (rank 1); the "two directions" are the two signs of one pattern. Antenna shift via `roll` was an unexplained magic number. | Kept (faithful to the original) but documented, `gram()` reports it, shift is `rfbd.antenna_shift`. |
| Hadamard | Hard-coded 128; truncation 128→114 breaks orthogonality in general (exactly orthogonal for rows 1,2). Row 1 is the Nyquist-rate pattern. | Order derived from `subcarriers`; `gram()` and the report log the realised cosine matrix. |
| Shapes | `antennas=3, subcarriers=114, packets=10` hard-coded in `train.py` and `evaluate_checkpoint.py` while the config had `data.input_shape`. | `build_trigger_from_config` derives them; `apply()` validates shape. |
| Mode map | Mode→trigger map duplicated in three files. | Single `MODE_TO_TRIGGER` in `attack/triggers/registry.py`. |
| `apply()` | Silent device/dtype copies, no validation of `eps`, shape, or tensor `k`. | Validated; cached device/dtype conversion. |

## 3. Hard-coded values removed

* All experimental defaults now live only in `configs/base_config.yaml`; `utils.config.resolve_config`
  merges every run on top of it. Removed contradictory literal fallbacks in `train.py`
  (`poison_rate 0.20` vs 0.01, `lam_trig_pose 5.0` vs 0.005, `latent_dim 256` vs 128,
  `lr 2e-3` vs 1e-3, `epochs 200` vs 50), the `AdamW`/"Adam" comment mismatch, `clip_grad_norm 5.0`
  (→ `training.grad_clip_norm`), `lam_ortho 1.0` (→ `attack.lam_ortho`).
* Fall-alarm thresholds were duplicated in `metrics.py` defaults, `generate_fall_labels.py` and the
  config → one `FallCriteria` dataclass built from the config; `vertical_axis` moved into it.
* State-collapse thresholds (0.80 / 0.85 / 0.40 / 20 cm) → `evaluation.state_collapse`.
* `BACKBONE, EPOCHS = "hpeli", 50` in `run_experiments.py` overrode the config → read from the config.
  Suite specs under `configs/experiments/` had ignored keys (`epochs`, `gate_threshold`, `output_file`,
  `train_rooms`, `default_params.epsilon/seed`) → removed or wired; default seed/loss-weight lists in code → removed.
* Skeleton indices/edges were duplicated in four modules → `utils/skeleton.py`.
* Dataset: `(3,114,10)`, `(17,3)`, "10 subjects per room", room tuple, subject→room mapping → config / `SCENE_SUBJECTS`.
* `plot_loss_ablation.py` contained the ablation numbers themselves and a `D:\…` output path → reads CSVs.
* "Prone fall"/"Lateral fall" labels and `k=0/k=1` assumptions → `fall_target_names(K)`, dynamic `K`.

## 4. Figures

New `figures/` package following the figures4papers `scientific-figure-making` skill (palette,
`apply_publication_style`, `finalize_figure` → png + pdf @ 300 dpi, semantic method colours):

* `figures/pipeline.py` — method-overview figure; trigger insets are the real Hadamard bank, skeletons
  are the real standing/fall templates, all numbers come from the config; the latent bars are labelled
  as loss targets, not measurements.
* `figures/plots.py` — latent distribution, 3-D skeleton demo, main benchmark bars with Wilson CIs
  (alarm and bone-consistent ASR), poison sweep, loss ablation, threshold sensitivity, trigger budget,
  trigger bank. Missing data is "n/a", colours bind to method identity, no twin axes.
* `utils/visualization.py` is a thin shim; dead helpers (`plot_gate_sweep`, orientation heuristics) and the
  import-time `matplotlib.use('Agg')` were removed.
* `scripts/build_paper_artifacts.py` builds the table and every figure from the CSVs.

## 5. What this audit cannot settle

* The alarm thresholds are a design choice. The sweep and false-alarm audit make the dependence
  visible; they do not validate the detector clinically. MM-Fi has no falls, so no real positive exists.
* Triggers are injected digitally after normalisation; physical realisability (hardware, channel,
  multipath, re-normalisation by the receiver) is untested.
* The vendored HPELI code hard-codes `dim2 = 10 / 8` pooled-shape arguments (`hpeli.py`); left untouched.
* Old results (`binary-fall-v4`) are not reproducible under the new protocol and must be regenerated.

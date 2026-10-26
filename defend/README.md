# `defend/` — defender-side analysis of the trained victim

Everything here is run by the **defender**, who owns the trained weights and a small **clean**
dataset, but does **not** know the trigger, the poisoned subset, or the attack target. The
detectors receive the poison mask *only to grade their own output* (AUROC / recall), never as a
defense input. Nothing in this package imports from `attack/` as a defense mechanism — it only
builds the trigger/poison plan to *score* detectability.

This folder is deliberately separate from the attack so the two threat sides never share code
or configuration.

## What it measures

| Module | Defense | Reference | Output |
|---|---|---|---|
| `detection.py` | Spectral Signatures | Tran, Li & Madry, NeurIPS 2018 | AUROC / recall / precision of detecting poisoned **training** samples |
| `detection.py` | Activation Clustering | Chen et al., SafeAI 2019 | 2-means silhouette, AUROC & recall of the poison cluster |
| `detection.py` | Linear-probe separability | — | cross-validated AUROC separating clean vs triggered **test** features (test-time readability of the trigger) |
| `model_repair.py` | Fine-Pruning | Liu, Dolan-Gavitt & Garg, RAID 2018 | ASR vs clean-MPJPE as dormant channels are pruned (+ optional fine-tune) |
| `model_repair.py` | Clean Fine-Tuning | — | ASR vs clean-MPJPE per clean epoch (on a deep copy) |

## Why these numbers matter for the latent attack

The latent loss *trains* `u_kᵀz` to separate triggered from clean activations. That is exactly
what a feature-space detector looks for, so **high** Spectral / Activation-Clustering / linear-probe
scores are the *expected, reportable cost* of the method — the attack buys input-space stealth and
pays with feature-space detectability. Report them honestly rather than claiming feature-space
stealth. Fine-Pruning and Fine-Tuning then test whether the backdoor survives standard clean-data
removal; an attack is only "robust" if ASR survives at a clean-utility cost the defender would
refuse to pay.

## Run

```bash
python -m defend.run_defenses \
    --checkpoint checkpoints/ours_latent_model.pt \
    --device cuda \
    --output_dir outputs/defenses \
    --defender_frames 2000 \   # clean training frames the defender owns (Fine-Pruning / Fine-Tuning set)
    --eval_frames 8000         # held-out test frames used to measure ASR
```

Writes `feature_space_detection.csv`, `fine_pruning.csv`, `fine_tuning.csv` and
`defense_summary.csv` to `--output_dir`.

"""
defend/detection.py
===================
Feature-space detectability of the backdoor.

The defender never sees the trigger or the poisoned subset; a detector produces a per-sample
score and the known poison mask is used **only** to grade that score (AUROC / recall), the way a
benchmark grades a detector.  All scores here are unsupervised w.r.t. the poison label.

* :func:`activation_clustering_audit` — Activation Clustering (Chen et al., SafeAI 2019):
  2-means on the (whitened) penultimate features; a poisoned class splits into a large clean
  cluster and a small poisoned one.  Reported: silhouette of the 2-split and how well the
  smaller cluster matches the poison mask (AUROC of the cluster assignment, poison recall).

* :func:`linear_probe_separability` — cross-validated logistic-regression AUROC separating
  *clean* from *triggered* test features.  This is the test-time question "is the trigger
  linearly readable off the latent?"; the latent attack maximises exactly this, so a high value
  is the expected, reportable cost of the method rather than a failure of it.

:func:`feature_space_report` bundles both with the projection separation statistics.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# Spectral Signatures already lives with the other stealth audits; re-export so a defender has a
# single import surface and the method is never reimplemented in two places.
from utils.defenses import spectral_signature_audit  # noqa: F401

_NAN = float("nan")


def activation_clustering_audit(
    features: torch.Tensor,
    poison_mask: torch.Tensor,
    *,
    n_components: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """2-means Activation Clustering on training features ``(N, m)``.

    ``features`` are the penultimate activations of the whole (one-class) training set; the
    smaller of the two clusters is the candidate poison cluster.  ``poison_mask`` grades the
    result only.  ``n_components`` PCA-whitens before clustering (Chen et al. reduce to a few
    independent components first).  Returns the silhouette of the split, the AUROC of using the
    smaller-cluster indicator as a poison score, poison recall of the smaller cluster, and its
    size fraction.
    """
    feats = features.double().cpu().numpy()
    mask = poison_mask.bool().cpu().numpy()
    n, m = feats.shape
    n_poison = int(mask.sum())
    if n_poison in (0, n) or n < 4:
        return {k: _NAN for k in ("silhouette", "auroc", "recall", "cluster_fraction")}

    centred = feats - feats.mean(0, keepdims=True)
    k = int(min(n_components, m, n - 1))
    # PCA via SVD (independent components, as in the reference) before 2-means.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    reduced = centred @ vt[:k].T

    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(reduced)
    # The smaller cluster is the candidate poison cluster.
    small = 0 if (labels == 0).sum() <= (labels == 1).sum() else 1
    is_small = labels == small
    sil = float(silhouette_score(reduced, labels)) if len(np.unique(labels)) > 1 else _NAN
    # AUROC of the binary cluster indicator against the true poison label.
    auroc = float(roc_auc_score(mask.astype(int), is_small.astype(float)))
    recall = 100.0 * float((is_small & mask).sum()) / n_poison
    return {
        "silhouette": sil,
        "auroc": max(auroc, 1.0 - auroc),  # cluster identity is unordered; report the informative side
        "recall": recall,
        "cluster_fraction": float(is_small.mean()),
    }


def linear_probe_separability(
    z_clean: torch.Tensor,
    z_trig: torch.Tensor,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-validated AUROC of a logistic probe separating clean from triggered features.

    ``z_clean`` / ``z_trig`` are ``(N, m)`` test-time penultimate features.  A high AUROC means
    the trigger is linearly readable off the latent (not feature-space stealthy).  Standardised
    features; stratified K-fold out-of-fold predictions so the AUROC is not optimistic.
    """
    zc = z_clean.double().cpu().numpy()
    zt = z_trig.double().cpu().numpy()
    if len(zc) < n_splits or len(zt) < n_splits:
        return {"auroc": _NAN, "n": float(len(zc) + len(zt))}
    X = np.concatenate([zc, zt], 0)
    y = np.concatenate([np.zeros(len(zc)), np.ones(len(zt))])
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc = float(roc_auc_score(y, proba))
    # Separability is label-symmetric: an out-of-fold AUROC below 0.5 is the same evidence of a
    # separating direction as its reflection, so report the magnitude (0.5 = indistinguishable).
    return {"auroc": max(auroc, 1.0 - auroc), "n": float(len(y))}


def feature_space_report(
    train_features: torch.Tensor,
    poison_mask: torch.Tensor,
    z_clean_test: torch.Tensor,
    z_trig_test: torch.Tensor,
    *,
    spectral_components: tuple[int, ...] = (1, 2),
    removal_factor: float = 1.5,
    seed: int = 0,
) -> dict[str, float]:
    """Full feature-space detectability report (training detection + test-time separability)."""
    out: dict[str, float] = {}
    for n in spectral_components:
        stats = spectral_signature_audit(train_features, poison_mask,
                                         n_components=int(n), removal_factor=removal_factor)
        out.update({f"spectral_{m}_c{n}": v for m, v in stats.items()})
    out.update({f"actclust_{k}": v for k, v in
                activation_clustering_audit(train_features, poison_mask, seed=seed).items()})
    out.update({f"linprobe_{k}": v for k, v in
                linear_probe_separability(z_clean_test, z_trig_test, seed=seed).items()})
    return out

"""Standard backdoor defenses used to audit stealthiness claims.

* :func:`spectral_signature_audit` — Spectral Signatures (Tran, Li & Madry, NeurIPS 2018):
  score every *training* sample by its squared projection on the top singular directions
  of the centred penultimate features and flag the highest scores.  The proposed method
  explicitly trains poisoned latents to sit at ``u_k^T z >= margin`` while clean latents
  stay below ``clean_penalty_threshold`` — i.e. it *creates* the separable cluster this
  defense looks for.  Any stealth claim must therefore report this audit.  Pose regression
  has no classes, so the defense runs on the whole training set (one "class").
* :func:`subcarrier_smoothing` — moving average along the subcarrier axis, the most common
  CSI denoising step.  Sylvester rows 1 (+-+-) and 2 (++--) sum to zero over every window of
  2 and 4 consecutive subcarriers respectively, so a width-4 window cancels the *additive*
  pattern of both K=2 triggers.  Because ``BaseTrigger.apply`` renormalises each frame after
  injection, a per-frame offset/contrast change (~half the raw trigger amplitude) survives
  smoothing; any ASR left after smoothing may be keyed to that renormalisation artefact.
  Even widths pad asymmetrically (half-subcarrier shift).
* :func:`packet_smoothing` — the same filter along packets (targets temporal impulses).
* :func:`remove_static_component` — per-subcarrier temporal-mean removal.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def subcarrier_smoothing(x: torch.Tensor, width: int) -> torch.Tensor:
    """Moving average of width ``width`` along the subcarrier axis of ``(B, C, F, T)`` CSI.

    Replicate padding keeps the shape; the result stays inside the input's ``[min, max]``.
    ``width == 1`` is the identity.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if width == 1:
        return x
    B, C, Fs, T = x.shape
    flat = x.permute(0, 1, 3, 2).reshape(B * C * T, 1, Fs)
    flat = F.pad(flat, ((width - 1) // 2, width // 2), mode="replicate")
    out = F.avg_pool1d(flat, kernel_size=width, stride=1)
    return out.reshape(B, C, T, Fs).permute(0, 1, 3, 2).contiguous()


def remove_static_component(x: torch.Tensor, renorm_eps: float = 1e-8) -> torch.Tensor:
    """Subtract each subcarrier's mean over packets, then per-frame min-max renormalise.

    Static-path removal is standard in CSI sensing (the static environment carries no motion
    information).  A trigger that is constant over packets — such as the Hadamard bank, which
    is broadcast identically over ``T`` — is removed exactly by this step.
    """
    dynamic = x - x.mean(dim=3, keepdim=True)
    mn = dynamic.amin(dim=(1, 2, 3), keepdim=True)
    mx = dynamic.amax(dim=(1, 2, 3), keepdim=True)
    return (dynamic - mn) / (mx - mn + renorm_eps)


def packet_smoothing(x: torch.Tensor, width: int) -> torch.Tensor:
    """Moving average of width ``width`` along the packet axis of ``(B, C, F, T)`` CSI."""
    return subcarrier_smoothing(x.transpose(2, 3), width).transpose(2, 3).contiguous()


@torch.no_grad()
@torch.no_grad()  # without it every batch's autograd graph stays alive on the GPU (OOM on the full train set)
def extract_features(net, dataset, device, batch_size: int) -> torch.Tensor:
    """Penultimate features ``z`` of every item of ``dataset`` (in order), on the CPU."""
    net.eval()
    feats = []
    for start in range(0, len(dataset), batch_size):
        xs = torch.stack([dataset[i][0] for i in range(start, min(start + batch_size, len(dataset)))])
        _, z = net(xs.to(device))
        feats.append(z.float().cpu())
    return torch.cat(feats)


def spectral_signature_audit(
    features: torch.Tensor,
    poison_mask: torch.Tensor,
    *,
    n_components: int,
    removal_factor: float,
) -> dict[str, float]:
    """Detection quality of Spectral Signatures against the known poisoned subset.

    Returns the AUROC of the outlier score, and the recall / precision of the standard
    removal rule (drop the ``removal_factor * n_poisoned`` highest-scoring samples).

    Limitations (report them): the removal rule uses the true poison count (oracle), which
    caps precision at ``1 / removal_factor``; without class conditioning the top directions
    are dominated by pose variance, so a *low* AUROC is weak evidence of stealth (a 1 %
    cluster offset by 0.5 in 3 dimensions scores ~0.51).  A *high* AUROC is strong evidence
    against stealth.
    """
    feats = features.double()
    mask = poison_mask.bool().cpu().numpy()
    n_poison = int(mask.sum())
    if n_poison == 0 or n_poison == len(mask):
        nan = float("nan")
        return {"auroc": nan, "recall": nan, "precision": nan}
    centred = feats - feats.mean(dim=0, keepdim=True)
    # Eigen-decomposition of the m x m covariance == right singular vectors of the data.
    _, vecs = torch.linalg.eigh(centred.T @ centred)
    top = vecs[:, -n_components:]
    scores = (centred @ top).pow(2).sum(dim=1).numpy()
    n_remove = min(len(scores), int(round(removal_factor * n_poison)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:n_remove]] = True
    caught = int((flagged & mask).sum())
    return {
        "auroc": float(roc_auc_score(mask.astype(int), scores)),
        "recall": 100.0 * caught / n_poison,
        "precision": 100.0 * caught / max(n_remove, 1),
    }

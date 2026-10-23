"""Measured (never asserted) properties of the triggered CSI.

``trigger_report`` runs the real :meth:`BaseTrigger.apply` on real frames and reports, per
trigger direction and on average:

* ``l2_rel_pct``  — ``100 * ||x_trig - x||_2 / ||x||_2``
* ``linf``        — ``max |x_trig - x|`` (CSI is in [0, 1], so this is the fraction of the dynamic range)
* ``psnr_db``     — ``10 log10(1 / MSE)`` with data range 1
* ``max_abs_cos`` — largest off-diagonal ``|cosine|`` of the trigger bank (``nan`` for K = 1)

These replace the earlier hard-coded claim that the perturbation "is ~0.4 %".
"""

from __future__ import annotations

import math

import torch

from attack.triggers import BaseTrigger


@torch.no_grad()
def trigger_report(trigger: BaseTrigger, x: torch.Tensor, eps: float) -> dict[str, float]:
    """Perturbation statistics of ``trigger`` on the CSI batch ``x`` of shape (B, C, F, T)."""
    x = x.detach().float().cpu()
    flat = lambda t: t.reshape(len(t), -1)
    x_norm = torch.linalg.vector_norm(flat(x), dim=1).clamp_min(1e-12)

    report: dict[str, float] = {}
    per_k: list[tuple[float, float, float]] = []
    for k in range(trigger.K):
        diff = trigger.apply(x, k, eps).cpu() - x
        l2 = torch.linalg.vector_norm(flat(diff), dim=1)
        mse = flat(diff).pow(2).mean(dim=1).clamp_min(1e-30)
        stats = (
            float((100.0 * l2 / x_norm).mean()),
            float(flat(diff).abs().amax(dim=1).mean()),
            float((10.0 * torch.log10(1.0 / mse)).mean()),
        )
        per_k.append(stats)
        report[f"trigger_l2_rel_pct_k{k}"], report[f"trigger_linf_k{k}"], report[f"trigger_psnr_db_k{k}"] = stats

    for i, name in enumerate(("l2_rel_pct", "linf", "psnr_db")):
        report[f"trigger_{name}"] = sum(s[i] for s in per_k) / len(per_k)

    if trigger.K > 1:
        gram = trigger.gram().cpu()
        off_diag = gram - torch.diag(torch.diag(gram))
        report["trigger_max_abs_cos"] = float(off_diag.abs().max())
    else:
        report["trigger_max_abs_cos"] = math.nan
    return report

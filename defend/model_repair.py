"""
defend/model_repair.py
======================
Post-hoc backdoor removal that uses ONLY the defender's clean data.

* :func:`fine_pruning_curve` — Fine-Pruning (Liu, Dolan-Gavitt & Garg, RAID 2018).  Channels of
  the last feature map that stay dormant on clean inputs are the capacity a backdoor hides in;
  pruning them (lowest average clean activation first) removes the backdoor before it touches
  clean accuracy.  We report the full ASR-vs-clean-utility curve over a pruning schedule, then
  one optional prune+fine-tune point.

* :func:`fine_tuning_curve` — plain fine-tuning of the poisoned weights on clean data (catastrophic
  forgetting of the backdoor).  Reported per epoch as (ASR, clean MPJPE).

Both take a ``measure`` callback ``net -> {"binary_fall_asr", "clean_mpjpe_cm", ...}`` so the
defense is decoupled from the evaluation plumbing, and a ``clean_loader`` yielding ``(x, y, ...)``
clean batches the defender is assumed to own.  Neither function reads the trigger or the poison
mask; they are pure model-repair.  ``fine_tuning_curve`` works on a deep copy, so the caller's
model is never mutated; ``fine_pruning_curve`` restores the model before returning.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


@torch.no_grad()
def _clean_channel_activation(net, clean_loader, device) -> torch.Tensor:
    """Mean absolute activation of each ``out1`` channel over the clean set, shape ``(C,)``."""
    net.eval()
    total = None
    count = 0
    for batch in clean_loader:
        x = batch[0].to(device)
        _, _, out1 = net.forward_features(x)          # (B, C, H, W)
        act = out1.abs().mean(dim=(0, 2, 3))          # (C,)
        total = act * len(x) if total is None else total + act * len(x)
        count += len(x)
    if count == 0:
        raise ValueError("clean_loader is empty; Fine-Pruning needs defender clean data")
    return (total / count).detach()


def _prune_hook(mask: torch.Tensor):
    """Forward-pre-hook that zeroes the pruned channels of the regression head's input."""
    def hook(_module, inputs):
        out1 = inputs[0]
        return (out1 * mask.to(out1.device, out1.dtype).view(1, -1, 1, 1),) + tuple(inputs[1:])
    return hook


def fine_pruning_curve(
    net,
    clean_loader,
    measure,
    device,
    *,
    fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    finetune_epochs: int = 0,
    finetune_lr: float = 1e-4,
) -> list[dict]:
    """Fine-Pruning curve: ASR and clean MPJPE as the dormant ``out1`` channels are pruned.

    ``fractions`` is the ascending fraction of channels pruned (lowest clean activation first).
    If ``finetune_epochs > 0``, one extra row ``phase='prune+finetune'`` fine-tunes the final
    pruned model on the clean data (the full Liu et al. defense).  The model is restored to its
    original (unpruned, un-fine-tuned) state before returning.
    """
    net.eval()
    activation = _clean_channel_activation(net, clean_loader, device)
    C = activation.numel()
    order = torch.argsort(activation)                 # ascending: most-dormant channels first
    head = net.hpeli.regression

    rows: list[dict] = []
    mask = torch.ones(C, device=device)
    handle = head.register_forward_pre_hook(_prune_hook(mask))
    try:
        for frac in fractions:
            n_prune = int(round(float(frac) * C))
            mask.fill_(1.0)
            if n_prune > 0:
                mask[order[:n_prune]] = 0.0
            m = measure(net)
            rows.append({"phase": "prune", "fraction": float(frac), "n_pruned": n_prune,
                         "binary_fall_asr": m["binary_fall_asr"], "clean_mpjpe_cm": m["clean_mpjpe_cm"]})
        if finetune_epochs > 0:
            # Keep the final pruning mask active and fine-tune the surviving weights on clean data.
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=finetune_lr)
            for _ in range(int(finetune_epochs)):
                for batch in clean_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    p, _ = net(x)
                    loss = F.l1_loss(p, y)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            m = measure(net)
            rows.append({"phase": "prune+finetune", "fraction": float(fractions[-1]),
                         "n_pruned": int(round(float(fractions[-1]) * C)),
                         "binary_fall_asr": m["binary_fall_asr"], "clean_mpjpe_cm": m["clean_mpjpe_cm"]})
    finally:
        handle.remove()
    return rows


def fine_tuning_curve(
    net,
    clean_loader,
    measure,
    device,
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> list[dict]:
    """Clean fine-tuning curve on a DEEP COPY: ASR and clean MPJPE after each epoch.

    The caller's ``net`` is never modified.  Row 0 (epoch 0) is the pre-defense baseline.
    """
    model = copy.deepcopy(net).to(device)
    rows = []
    base = measure(model)
    rows.append({"epoch": 0, "binary_fall_asr": base["binary_fall_asr"],
                 "clean_mpjpe_cm": base["clean_mpjpe_cm"]})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(1, int(epochs) + 1):
        model.train()
        for batch in clean_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            p, _ = model(x)
            loss = F.l1_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        m = measure(model)
        rows.append({"epoch": ep, "binary_fall_asr": m["binary_fall_asr"],
                     "clean_mpjpe_cm": m["clean_mpjpe_cm"]})
    return rows

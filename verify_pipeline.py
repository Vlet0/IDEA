"""Fast structural verification of the canonical protocol (no dataset needed)."""

import argparse
from pathlib import Path

import torch

from attack import (MODE_TO_TRIGGER, FTrojanLoss, LatentBackdoorLoss, build_poison_plan,
                    build_trigger_from_config, compute_target_poses, get_semantic_fall_targets,
                    get_semantic_standing_pose, make_U)
from models.backbone import build_backbone
from run_experiments import _config
from train import _build_criterion, load_config
from utils.config import resolve_config
from utils.metrics import (FallCriteria, compute_binary_fall_asr, compute_latent_separation_stats,
                           compute_mpjpe, is_biomechanical_fall)


def verify(device: torch.device) -> None:
    root = Path(__file__).resolve().parent
    cfg = resolve_config(load_config(str(root / "configs" / "base_config.yaml")))
    antennas, subcarriers, packets = cfg["data"]["input_shape"]
    K, m = cfg["attack"]["K"], cfg["model"]["latent_dim"]
    assert cfg["model"]["backbone"] == "hpeli"
    assert "trigger_type" not in cfg["attack"]

    model = build_backbone(cfg["model"]["backbone"], m=m, num_joints=cfg["model"]["num_joints"],
                           subcarrier_num=subcarriers).to(device)
    x = torch.rand(2, antennas, subcarriers, packets, device=device)
    with torch.no_grad():
        pose, latent = model(x)
    assert pose.shape == (2, cfg["model"]["num_joints"], 3)
    assert latent.shape == (2, m)

    for mode in MODE_TO_TRIGGER:
        run_cfg = _config(cfg, mode, f"verification_{mode}")
        assert run_cfg["attack"]["mode"] == mode and "trigger_type" not in run_cfg["attack"]
        trigger = build_trigger_from_config(run_cfg, device=device)
        output = trigger.apply(x, 0, eps=cfg["attack"]["epsilon"])
        assert output.shape == x.shape and torch.isfinite(output).all()
        assert output.amin() >= 0 and output.amax() <= 1
        # every trigger family honours the shared per-antenna unit-L2 budget
        norms = torch.linalg.vector_norm(trigger.patterns, dim=(2, 3))
        assert (norms <= 1.0 + 1e-4).all()
        assert torch.linalg.vector_norm(trigger.patterns, dim=(2, 3)).amax() > 0.99

    plan = build_poison_plan(1000, cfg["attack"]["poison_rate"], K, seed=0)
    assert plan.n_poisoned == round(1000 * cfg["attack"]["poison_rate"])
    counts = plan.per_trigger_counts(K)
    assert sum(counts) == plan.n_poisoned and max(counts) - min(counts) <= 1  # balanced directions

    ps = 0.3
    targets = get_semantic_fall_targets(K=K, ps=ps, device=device)
    mask = torch.tensor([True, False], device=device)
    indices = torch.tensor([0, 1], device=device)
    target_poses = compute_target_poses(pose.detach(), indices, targets, **cfg["attack"]["target"])
    U = make_U(m=m, K=K, seed=0, device=device)
    for mode, loss in (("latent", _build_criterion("latent", cfg["attack"])),
                       ("ftrojan", FTrojanLoss())):
        p = pose.detach().clone().requires_grad_(True)
        if isinstance(loss, LatentBackdoorLoss):
            value, _ = loss(p, latent.detach(), pose.detach(), U, mask, indices, target_poses)
        else:
            value, _ = loss(p, pose.detach(), mask, target_poses)
        assert torch.isfinite(value), mode

    criteria = FallCriteria.from_dict(cfg["evaluation"]["fall_criteria"])
    upright = get_semantic_standing_pose(ps=ps).unsqueeze(0).repeat(2, 1, 1)
    fallen = targets.cpu().clone()
    assert not is_biomechanical_fall(upright, ps, criteria).any()
    assert is_biomechanical_fall(fallen, ps, criteria).all()
    binary = compute_binary_fall_asr(upright, fallen, ps, ground_truth_poses=upright, criteria=criteria)
    assert binary["binary_fall_asr"] == 100.0 and binary["n_success"] == 2
    gate = cfg["attack"]["gate_threshold"]
    latent_stats = compute_latent_separation_stats(
        torch.tensor([[0.1, 0.2], [0.2, 0.3]]), torch.tensor([[2.0, 3.0], [2.5, 4.0]]), gate_thresh=gate)
    assert latent_stats["auroc"] == 1.0
    assert latent_stats["false_activation_rate_pct"] == 0.0
    assert latent_stats["true_activation_rate_pct"] == 100.0
    assert compute_mpjpe(fallen, fallen, ps).max() == 0
    print("Protocol verification passed: HPELI, Hadamard + FTrojan triggers, poisoning plan and loss branches are valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    verify(torch.device(parser.parse_args().device))

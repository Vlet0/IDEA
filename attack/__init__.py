"""Attack components for the three preregistered experimental methods."""

from .latent_attack import (TARGET_TYPES, build_semantic_target_bank, compose_trajectory_pose,
                            compute_target_poses, fall_target_names, get_curled_fall_pose,
                            get_fall_trajectory, get_semantic_fall_pose, get_semantic_fall_targets,
                            get_semantic_standing_pose, is_fall_target, make_U, make_U_from_latents,
                            project_latent, target_bank_names)
from .losses import FTrojanLoss, LatentBackdoorLoss, PoisonedERMLoss, compute_causal_losses
from .poisoning import (PoisonedDataset, PoisonPlan, build_poison_plan, dataset_tensors,
                        validate_poison_rate)
from .triggers import (MODE_TO_TRIGGER, BaseTrigger, FTrojanTrigger, HadamardTrigger,
                       apply_trigger, build_trigger, build_trigger_from_config,
                       list_triggers, trigger_for_mode)

__all__ = [
    "BaseTrigger", "HadamardTrigger", "FTrojanTrigger",
    "MODE_TO_TRIGGER", "build_trigger", "build_trigger_from_config", "trigger_for_mode",
    "list_triggers", "apply_trigger", "make_U",
    "get_semantic_fall_pose", "get_semantic_fall_targets", "get_semantic_standing_pose",
    "fall_target_names", "compute_target_poses", "project_latent",
    "TARGET_TYPES", "build_semantic_target_bank", "target_bank_names", "is_fall_target",
    "get_curled_fall_pose", "get_fall_trajectory", "compose_trajectory_pose",
    "LatentBackdoorLoss", "PoisonedERMLoss", "FTrojanLoss", "compute_causal_losses",
    "make_U_from_latents",
    "PoisonPlan", "PoisonedDataset", "build_poison_plan", "dataset_tensors", "validate_poison_rate",
]

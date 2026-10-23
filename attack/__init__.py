"""Attack components for the three preregistered experimental methods."""

from .latent_attack import (compute_target_poses, fall_target_names, get_semantic_fall_pose,
                            get_semantic_fall_targets, get_semantic_standing_pose,
                            make_U, project_latent)
from .losses import FTrojanLoss, LatentBackdoorLoss, PoisonedERMLoss
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
    "LatentBackdoorLoss", "PoisonedERMLoss", "FTrojanLoss",
    "PoisonPlan", "PoisonedDataset", "build_poison_plan", "dataset_tensors", "validate_poison_rate",
]

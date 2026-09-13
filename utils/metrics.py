import numpy as np
import torch

def compute_mpjpe(pred, target, ps=1.0):
    """
    Mean Per Joint Position Error (in centimeters).
    pred, target: (B, 17, 3) normalized pose tensors.
    ps: pose scale factor in meters.
    Returns: 1D tensor of errors per frame in centimeters.
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    # L2 distance across the 3 spatial coordinates, then average across all 17 joints
    err_normalized = torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1)
    err_cm = err_normalized * ps * 100.0
    return err_cm

def compute_asr(atk_err_cm, threshold_cm=15.0):
    """
    Attack Success Rate (% of triggered samples with MPJPE < threshold against target pose).
    """
    if isinstance(atk_err_cm, torch.Tensor):
        return float((atk_err_cm < threshold_cm).float().mean().item() * 100.0)
    else:
        return float(np.mean(atk_err_cm < threshold_cm) * 100.0)

def compute_gate_fpr(clean_fired):
    """
    False positive rate of the backdoor gate detector on clean frames.
    clean_fired: boolean / float tensor indicating gate activation on clean input.
    """
    if isinstance(clean_fired, torch.Tensor):
        return float(clean_fired.float().mean().item() * 100.0)
    else:
        return float(np.mean(clean_fired) * 100.0)

def compute_detector_auroc(s_clean_norm, s_trig_norm):
    """
    Computes AUROC for separating clean activations from triggered activations based on latent norm.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.zeros(len(s_clean_norm)), np.ones(len(s_trig_norm))])
    y_scores = np.concatenate([s_clean_norm, s_trig_norm])
    return float(roc_auc_score(y_true, y_scores))

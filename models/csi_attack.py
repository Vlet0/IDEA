import torch

def apply_csi_trigger(x, k, csi_trig, eps=0.15):
    """
    Injects CSI-space trigger before re-applying MM-Fi min-max normalization.
    Preserves exact notebook logic:
      x = x + eps * CSI_TRIG[k] * x.amax(dim=(1,2,3), keepdim=True)
      mn = x.amin(dim=(1,2,3), keepdim=True)
      mx = x.amax(dim=(1,2,3), keepdim=True)
      return (x - mn) / (mx - mn + 1e-8)
      
    Args:
        x: CSI tensor (B, 3, 114, 10)
        k: Trigger index (integer or 1D tensor of indices)
        csi_trig: Pre-computed normalized Hadamard triggers (K, 3, 114, 10)
        eps: Perturbation amplitude scale (default 0.15)
    """
    if isinstance(k, int):
        trigger_pattern = csi_trig[k]
    else:
        # k is a tensor of indices (B,)
        trigger_pattern = csi_trig[k]
        
    x = x + eps * trigger_pattern * x.amax(dim=(1, 2, 3), keepdim=True)
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - mn) / (mx - mn + 1e-8)

import torch
from scipy.linalg import hadamard

def generate_hadamard_csi_triggers(K=2, subcarriers=114, antennas=3, packets=10, device='cpu'):
    """
    Constructs channel-orthogonal CSI triggers from Hadamard matrix rows.
    Rows [1, 2, ..., K] are used to avoid the all-ones DC row (row 0),
    which is cancelled by per-frame min-max normalization.
    
    Returns:
        CSI_TRIG: tensor of shape (K, antennas, subcarriers, packets) normalized to unit L2 norm.
    """
    # Size must be a power of 2, 128 >= 114
    Hm = hadamard(128)
    
    # Avoid row 0 (all ones), select rows 1 .. K
    selected_rows = list(range(1, K + 1))
    H = torch.tensor(Hm[selected_rows, :subcarriers], dtype=torch.float32)
    
    CSI_TRIG = torch.zeros(K, antennas, subcarriers, packets, dtype=torch.float32)
    for k in range(K):
        CSI_TRIG[k] = H[k].view(1, subcarriers, 1).expand(antennas, subcarriers, packets)
        
    # Normalize each trigger to unit Frobenius / vector norm
    norm = torch.linalg.vector_norm(CSI_TRIG, dim=(1, 2, 3), keepdim=True)
    CSI_TRIG = CSI_TRIG / norm
    
    return CSI_TRIG.to(device)

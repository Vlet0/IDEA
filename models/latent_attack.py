import torch
import torch.nn.functional as F

def make_U(m=256, K=2, seed=0, device='cpu'):
    """
    Generates an orthonormal basis U in latent space via QR decomposition.
    U: (m, K), satisfying U^T U = I_K.
    """
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(m, K, generator=g))
    return Q.to(device)

def get_deformation_matrix(K=2, ps=1.0, shift_m=0.30, device='cpu'):
    """
    Defines K rigid translation directions in normalized pose space.
    - Direction 0: +shift_m along X axis
    - Direction 1: -shift_m along Z axis
    Preserves exact bone lengths (C4 Kinematic constraint).
    """
    D = torch.zeros(K, 17, 3, dtype=torch.float32)
    normalized_shift = shift_m / ps
    
    # Notebook default 2 directions
    D[0, :, 0] =  normalized_shift
    if K > 1:
        D[1, :, 2] = -normalized_shift
        
    # Generalized pattern if K > 2 (for sweeps/ablations)
    if K > 2:
        directions = [(1, 0), (0, -1), (2, 1), (1, -1), (0, 1), (2, -1)]
        for k in range(2, K):
            axis, sign = directions[(k - 2) % len(directions)]
            D[k, :, axis] = sign * normalized_shift
            
    return D.to(device)

def apply_latent_deformation(p, z, U, D, gate=2.0, temp=0.5, gate_mode='max_abs'):
    """
    Applies latent gating and adds deformation field:
    s = z @ U
    m = (||s|| or max|s|) >= gate
    w = m * softmax(s / temp)
    p_deformed = p + sum_k w_k D_k
    """
    s = z @ U  # (B, K)
    
    if gate_mode == 'max_abs':
        m = s.abs().amax(dim=1, keepdim=True) >= gate
    else:  # vector_norm
        m = torch.linalg.vector_norm(s, dim=1, keepdim=True) >= gate
        
    w = m.float() * F.softmax(s / temp, dim=1)  # (B, K)
    p_deformed = p + torch.einsum('bk,kjc->bjc', w, D)
    
    return p_deformed, s, m.squeeze(1)

import sys
import torch
import numpy as np

def run_tests():
    print("=== TEST 1: Model Architecture ===")
    from models.backbone import Net
    net = Net(m=256)
    x = torch.randn(4, 3, 114, 10)
    p, z = net(x)
    assert p.shape == (4, 17, 3), f"Pose shape mismatch: {p.shape}"
    assert z.shape == (4, 256), f"Latent shape mismatch: {z.shape}"
    print("Model forward pass: PASSED")

    print("\n=== TEST 2: Latent Trigger & Orthogonality ===")
    from models.latent_attack import make_U, get_deformation_matrix, apply_latent_deformation
    U = make_U(m=256, K=2, seed=0)
    diff = (U.T @ U) - torch.eye(2)
    assert diff.abs().max().item() < 1e-6, f"U is not orthogonal: {diff}"
    print("Orthogonality U^T U = I: PASSED")

    D = get_deformation_matrix(K=2, ps=0.30, shift_m=0.30)
    assert D.shape == (2, 17, 3), f"Deformation shape mismatch: {D.shape}"
    assert torch.isclose(D[0, :, 0], torch.tensor(1.0)).all(), "D[0] shift mismatch"
    assert torch.isclose(D[1, :, 2], torch.tensor(-1.0)).all(), "D[1] shift mismatch"
    print("Deformation matrix D: PASSED")

    print("\n=== TEST 3: Kinematic Constraint C4 (Bone Length Preservation) ===")
    from datasets.transforms import compute_bone_length_error
    clean_pose = np.random.randn(17, 3)
    deformed_pose = clean_pose + D[0].numpy() * 0.30
    bone_err = compute_bone_length_error(clean_pose, deformed_pose)
    assert bone_err < 1e-4, f"Bone length error too large: {bone_err}"
    print(f"Bone length preservation error: {bone_err:.8f} cm: PASSED")

    print("\n=== TEST 4: CSI Hadamard Trigger & Normalization ===")
    from utils.hadamard import generate_hadamard_csi_triggers
    from models.csi_attack import apply_csi_trigger
    CSI_TRIG = generate_hadamard_csi_triggers(K=2)
    assert CSI_TRIG.shape == (2, 3, 114, 10), f"Trigger shape mismatch: {CSI_TRIG.shape}"
    
    x_sample = torch.rand(4, 3, 114, 10)
    x_trig = apply_csi_trigger(x_sample.clone(), 0, CSI_TRIG, eps=0.15)
    assert x_trig.shape == x_sample.shape, "CSI trigger output shape mismatch"
    assert x_trig.amin() >= 0.0 and x_trig.amax() <= 1.0 + 1e-5, "Min-max normalization violated"
    print("CSI Hadamard trigger and normalization: PASSED")

    print("\n=== TEST 5: Loss Function ===")
    from models.losses import LatentBackdoorLoss, CSIBaselineLoss
    crit_lat = LatentBackdoorLoss(lam=2.0, margin=3.0, clean_thresh=0.5)
    pm = torch.tensor([True, False, True, False])
    kk = torch.tensor([0, 1, 1, 0])
    loss, info = crit_lat(p, z, p.clone(), U, pm, kk)
    assert torch.isfinite(loss), "Latent loss is not finite"
    print(f"Latent loss computed successfully: {loss.item():.4f}: PASSED")

    crit_csi = CSIBaselineLoss()
    loss_csi, _ = crit_csi(p, p.clone(), D, pm, kk)
    assert torch.isfinite(loss_csi), "CSI loss is not finite"
    print(f"CSI baseline loss computed successfully: {loss_csi.item():.4f}: PASSED")

    print("\n==========================================")
    print("  ALL CORE LOGIC UNIT TESTS PASSED (5/5)  ")
    print("==========================================")

if __name__ == '__main__':
    run_tests()

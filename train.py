import os
import time
import argparse
import yaml
import torch
import numpy as np

from models import Net, make_U, get_deformation_matrix, apply_csi_trigger, LatentBackdoorLoss, CSIBaselineLoss
from utils.hadamard import generate_hadamard_csi_triggers
from utils.logger import setup_logger, save_dict_to_csv
from utils.visualization import plot_skeleton_demo, plot_latent_distribution, plot_gate_sweep
from datasets.mmfi_dataset import load_mmfi_data
from datasets.transforms import compute_bone_length_error
from evaluate import evaluate_model, evaluate_per_trigger, sweep_gate_thresholds

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(config):
    """
    Main training routine for Latent-Space Backdoor and CSI-space Baseline.
    Preserves 100% exact training logic of Untitled13.ipynb (train3 and train2).
    """
    seed = config.get('seed', 0)
    set_seed(seed)
    
    device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)

    output_dir = config.get('training', {}).get('output_dir', './outputs/')
    save_dir = config.get('training', {}).get('save_dir', './checkpoints/')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    exp_name = config.get('experiment_name', 'backdoor_exp')
    logger = setup_logger(output_dir, name=exp_name)
    logger.info(f"=== Starting Experiment: {exp_name} on device [{device}] ===")

    # 1. Load MM-Fi Dataset
    train_loader, test_loader, PS, meta = load_mmfi_data(config)
    logger.info(f"Loaded MM-Fi data successfully. Pose scale PS: {PS:.4f} m")

    # 2. Setup Attack Parameters
    atk_cfg = config.get('attack', {})
    mode = atk_cfg.get('mode', 'latent')  # 'latent' or 'csi'
    K = atk_cfg.get('K', 2)
    eps = atk_cfg.get('epsilon', 0.15)
    poison_rate = atk_cfg.get('poison_rate', 0.20)
    lam = atk_cfg.get('loss_lambda', 2.0)
    margin = atk_cfg.get('margin', 3.0)
    clean_thresh = atk_cfg.get('clean_penalty_threshold', 0.5)
    shift_m = atk_cfg.get('target_shift_m', 0.30)
    gate = atk_cfg.get('gate_threshold', 2.0)

    # 3. Build Orthonormal U, Deformation D, and CSI Hadamard Triggers
    latent_dim = config.get('model', {}).get('latent_dim', 256)
    U = make_U(m=latent_dim, K=K, seed=seed, device=device)
    D = get_deformation_matrix(K=K, ps=PS, shift_m=shift_m, device=device)
    CSI_TRIG = generate_hadamard_csi_triggers(K=K, device=device)

    logger.info(f"Attack mode: {mode.upper()} | K={K} | eps={eps} | poison_rate={poison_rate}")
    logger.info(f"Deformation shift: {shift_m} m (normalized: {shift_m / PS:.4f})")
    logger.info(f"U^T U orthogonality check:\n{(U.T @ U).cpu().numpy().round(4)}")

    # 4. Initialize Network, Optimizer, and OneCycleLR Scheduler
    net = Net(m=latent_dim).to(device)
    lr = config.get('training', {}).get('learning_rate', 2e-3)
    wd = config.get('training', {}).get('weight_decay', 1e-4)
    epochs = config.get('training', {}).get('epochs', 25)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)

    latent_criterion = LatentBackdoorLoss(lam=lam, margin=margin, clean_thresh=clean_thresh)
    csi_criterion = CSIBaselineLoss()

    # 5. Training Loop
    t0 = time.time()
    log_interval = config.get('training', {}).get('log_interval', 5)

    for ep in range(epochs):
        net.train()
        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            b = len(x_batch)
            x = x_batch.to(device)
            y = y_batch.to(device)

            # Sample poisoned indices
            pm = torch.rand(b, device=device) < poison_rate
            kk = torch.randint(0, K, (b,), device=device)

            # Apply CSI trigger to poisoned subset
            for k in range(K):
                s = pm & (kk == k)
                if s.any():
                    x[s] = apply_csi_trigger(x[s], k, CSI_TRIG, eps=eps)

            p, z = net(x)

            if mode == 'csi':
                loss, _ = csi_criterion(p, y, D, pm, kk)
            else:
                loss, _ = latent_criterion(p, z, y, U, pm, kk)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * b
            total_samples += b

        if (ep + 1) % log_interval == 0 or (ep + 1) == epochs:
            avg_loss = total_loss / max(total_samples, 1)
            elapsed = time.time() - t0
            logger.info(f"  Epoch [{ep+1:2d}/{epochs:2d}] | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.0f}s")

    # 6. Save Model Checkpoint
    ckpt_path = os.path.join(save_dir, f"{exp_name}_{mode}_model.pt")
    torch.save({
        'net_state_dict': net.state_dict(),
        'U': U.cpu(),
        'D': D.cpu(),
        'PS': PS,
        'config': config
    }, ckpt_path)
    logger.info(f"Saved checkpoint to: {ckpt_path}")

    # 7. Comprehensive Evaluation
    logger.info(f"\n===== EVALUATION: {mode.upper()} =====")
    eval_res = evaluate_model(net, test_loader, mode, U, D, CSI_TRIG, PS, device=device, gate=gate, eps=eps)

    logger.info(f"  Clean MPJPE        : {eval_res['clean_mpjpe_cm']:6.1f} cm")
    logger.info(f"  Error vs TARGET    : {eval_res['err_target_cm']:6.1f} cm")
    logger.info(f"  Error vs ORIGINAL  : {eval_res['err_original_cm']:6.1f} cm")
    logger.info(f"  ASR (<15cm)        : {eval_res['asr']:5.1f} %")
    logger.info(f"  Gate False Positive: {eval_res['clean_fp_rate']:5.2f} %")
    logger.info(f"  ||s|| clean={eval_res['s_clean_norm']:.2f} | trig={eval_res['s_trig_norm']:.2f}")

    # 8. Per-Trigger Analysis & Gate Sweeping (Latent mode)
    if mode == 'latent':
        per_trig = evaluate_per_trigger(net, test_loader, U, D, CSI_TRIG, PS, device=device, gate=gate, eps=eps)
        sweep_data, best_gate = sweep_gate_thresholds(net, test_loader, U, D, CSI_TRIG, PS, device=device)

        # Plot latent distribution
        sc_np = eval_res['s_clean'].numpy()
        st_np = eval_res['s_trig'].numpy()
        plot_latent_distribution(sc_np, st_np, gate=gate, output_path=os.path.join(output_dir, f"{exp_name}_latent_dist.png"))

        # Plot gate sweep curve
        gates = [s['gate'] for s in sweep_data]
        asrs = [s['asr'] for s in sweep_data]
        fprs = [s['fp'] for s in sweep_data]
        mpjpes = [s['clean_mpjpe'] for s in sweep_data]
        plot_gate_sweep(gates, asrs, fprs, mpjpes, output_path=os.path.join(output_dir, f"{exp_name}_gate_sweep.png"))

    # 9. 3D Skeleton Visualization & Bone Length Verification (Sample 500)
    logger.info("\n--- Validating Kinematic Constraint C4 (Bone Length Conservation) ---")
    net.eval()
    test_X = meta['Xte']
    test_Y = meta['Yte']
    sample_idx = min(500, len(test_X) - 1)

    with torch.no_grad():
        x_clean = test_X[sample_idx:sample_idx+1].to(device)
        p_c, z_c = net(x_clean)
        clean_pose_m = (p_c[0].cpu().numpy()) * PS

        trig_poses_m = []
        trig_scores_list = []
        for k in range(K):
            x_k = apply_csi_trigger(x_clean.clone(), k, CSI_TRIG, eps=eps)
            p_k, z_k = net(x_k)
            s_k = z_k @ U
            if mode == 'latent':
                m_k = s_k.abs().amax(dim=1, keepdim=True) >= gate
                w_k = m_k.float() * torch.nn.functional.softmax(s_k / 0.5, dim=1)
                p_k = p_k + torch.einsum('bk,kjc->bjc', w_k, D)
            trig_poses_m.append(p_k[0].cpu().numpy() * PS)
            trig_scores_list.append(s_k[0].cpu().numpy())

        # Measure bone length preservation error (in cm)
        bone_err = compute_bone_length_error(clean_pose_m, clean_pose_m + (D[0].cpu().numpy() * PS))
        logger.info(f"Bone length deviation (pure rigid translation): {bone_err:.6f} cm")

        # Save 3D skeleton visualization
        skeleton_path = os.path.join(output_dir, f"{exp_name}_skeleton_demo.png")
        plot_skeleton_demo(clean_pose_m, trig_poses_m, trig_scores_list, output_path=skeleton_path)

    # 10. Record Results Summary
    summary_record = {
        'experiment_name': exp_name,
        'mode': mode,
        'seed': seed,
        'clean_mpjpe_cm': eval_res['clean_mpjpe_cm'],
        'err_target_cm': eval_res['err_target_cm'],
        'err_original_cm': eval_res['err_original_cm'],
        'asr': eval_res['asr'],
        'clean_fp_rate': eval_res['clean_fp_rate'],
        's_clean_norm': eval_res['s_clean_norm'],
        's_trig_norm': eval_res['s_trig_norm']
    }
    save_dict_to_csv([summary_record], os.path.join(output_dir, "experiment_summary.csv"))
    logger.info(f"Experiment {exp_name} completed successfully!\n")
    return summary_record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train WiFi-CSI Latent Backdoor or Baseline.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help="Path to YAML config file.")
    parser.add_argument('--mode', type=str, choices=['latent', 'csi'], default=None,
                        help="Override attack mode (latent or csi).")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override random seed.")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Override batch size.")
    parser.add_argument('--exp_name', type=str, default=None,
                        help="Override experiment name.")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Handle defaults inheritance if specified
    if 'defaults' in config:
        base_path = config['defaults']
        if os.path.exists(base_path):
            with open(base_path, 'r', encoding='utf-8') as f:
                base_cfg = yaml.safe_load(f)
            # Recursive update
            for k, v in config.items():
                if isinstance(v, dict) and k in base_cfg:
                    base_cfg[k].update(v)
                else:
                    base_cfg[k] = v
            config = base_cfg

    # Apply CLI overrides
    if args.mode:
        config.setdefault('attack', {})['mode'] = args.mode
    if args.seed is not None:
        config['seed'] = args.seed
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.exp_name:
        config['experiment_name'] = args.exp_name

    train(config)

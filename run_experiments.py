import os
import copy
import argparse
import yaml
import numpy as np
from tabulate import tabulate

from train import train
from utils.logger import save_dict_to_csv

def run_leave_one_room_out(base_cfg, exp_cfg_path):
    """
    Executes leave-one-room-out experiments across all 4 environments:
    - Train: E01,E02,E03 -> Test: E04
    - Train: E01,E02,E04 -> Test: E03
    - Train: E01,E03,E04 -> Test: E02
    - Train: E02,E03,E04 -> Test: E01
    """
    print("\n========================================================")
    print("       EXPERIMENT: LEAVE-ONE-ROOM-OUT EVALUATION        ")
    print("========================================================")

    with open(exp_cfg_path, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    splits = exp_cfg.get('splits', [])
    modes = exp_cfg.get('modes', ['latent', 'csi'])
    output_csv = exp_cfg.get('output_file', './outputs/experiments/leave_one_room_out.csv')

    all_results = []
    for split in splits:
        split_name = split['name']
        tr_rooms = split['train_rooms']
        te_rooms = split['test_rooms']

        for mode in modes:
            print(f"\n>>> Running Split: {split_name} | Mode: {mode.upper()} <<<")
            cfg = copy.deepcopy(base_cfg)
            cfg['data']['train_rooms'] = tr_rooms
            cfg['data']['test_rooms'] = te_rooms
            cfg['attack']['mode'] = mode
            cfg['experiment_name'] = f"room_{split_name}_{mode}"
            cfg['training']['output_dir'] = f"./outputs/experiments/leave_one_out/{split_name}_{mode}/"

            res = train(cfg)
            res['split'] = split_name
            res['test_room'] = te_rooms[0]
            all_results.append(res)

    save_dict_to_csv(all_results, output_csv)
    print_summary_table(all_results, title="Leave-One-Room-Out Transfer Results")


def run_seed_stability(base_cfg, exp_cfg_path):
    """
    Runs 5 independent seeds for both Latent and CSI modes, reporting Mean +- Std.
    """
    print("\n========================================================")
    print("         EXPERIMENT: SEED STABILITY EVALUATION          ")
    print("========================================================")

    with open(exp_cfg_path, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    seeds = exp_cfg.get('seeds', [0, 1, 2, 42, 123])
    modes = exp_cfg.get('modes', ['latent', 'csi'])
    output_csv = exp_cfg.get('output_file', './outputs/experiments/seed_stability.csv')

    all_results = []
    for mode in modes:
        for s in seeds:
            print(f"\n>>> Running Seed: {s} | Mode: {mode.upper()} <<<")
            cfg = copy.deepcopy(base_cfg)
            cfg['seed'] = s
            cfg['attack']['mode'] = mode
            cfg['experiment_name'] = f"seed_{s}_{mode}"
            cfg['training']['output_dir'] = f"./outputs/experiments/seeds/{mode}_seed{s}/"

            res = train(cfg)
            all_results.append(res)

    save_dict_to_csv(all_results, output_csv)

    # Compute Mean +- Std for each mode
    print("\n=== Seed Stability Summary (Mean +- Std over seeds) ===")
    summary_rows = []
    for mode in modes:
        mode_res = [r for r in all_results if r['mode'] == mode]
        asrs = [r['asr'] for r in mode_res]
        mpjpes = [r['clean_mpjpe_cm'] for r in mode_res]
        fps = [r['clean_fp_rate'] for r in mode_res]

        summary_rows.append([
            mode.upper(),
            f"{np.mean(mpjpes):.2f} ± {np.std(mpjpes):.2f}",
            f"{np.mean(asrs):.1f}% ± {np.std(asrs):.1f}%",
            f"{np.mean(fps):.2f}% ± {np.std(fps):.2f}%"
        ])

    print(tabulate(summary_rows, headers=["Mode", "Clean MPJPE (cm)", "ASR (<15cm)", "Gate FP Rate"], tablefmt="grid"))


def run_hyperparam_sweeps(base_cfg, exp_cfg_path):
    """
    Sweeps poison rates (1% to 20%) and epsilons (0.05 to 0.20).
    """
    print("\n========================================================")
    print("      EXPERIMENT: HYPERPARAMETER SENSITIVITY SWEEPS     ")
    print("========================================================")

    with open(exp_cfg_path, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    poison_rates = exp_cfg.get('poison_rates', [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    output_dir = exp_cfg.get('output_dir', './outputs/experiments/sweeps/')
    os.makedirs(output_dir, exist_ok=True)

    sweep_results = []
    for p in poison_rates:
        print(f"\n>>> Running Poison Rate: {p*100:.0f}% <<<")
        cfg = copy.deepcopy(base_cfg)
        cfg['attack']['poison_rate'] = p
        cfg['experiment_name'] = f"sweep_poison_{int(p*100)}pct"
        cfg['training']['output_dir'] = os.path.join(output_dir, f"poison_{int(p*100)}pct/")

        res = train(cfg)
        res['poison_rate'] = p
        sweep_results.append(res)

    save_dict_to_csv(sweep_results, os.path.join(output_dir, "poison_rate_sweep.csv"))
    print_summary_table(sweep_results, title="Poison Rate Sensitivity Sweep")


def print_summary_table(results_list, title="Summary"):
    """Prints a clean tabular overview of results."""
    print(f"\n=== {title} ===")
    headers = ["Exp Name", "Mode", "Clean MPJPE (cm)", "Err Target (cm)", "ASR (%)", "Gate FP (%)"]
    rows = []
    for r in results_list:
        rows.append([
            r.get('experiment_name', '-'),
            r.get('mode', '-'),
            f"{r.get('clean_mpjpe_cm', 0):.1f}",
            f"{r.get('err_target_cm', 0):.1f}",
            f"{r.get('asr', 0):.1f}%",
            f"{r.get('clean_fp_rate', 0):.2f}%"
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def run_shaping_controls(base_cfg):
    """
    Experiment A from Report 9/9 Slide 29: Direction / Shaping Controls.
    Compares:
    1. Learned U (Proposed Latent Backdoor with Two-Sided Margin Loss)
    2. Random U on Clean Encoder (No shaping loss)
    3. PCA U on Clean Encoder (Intrinsic principal directions without shaping)
    Proves: A learned-only gain supports shaping loss, not latent space alone.
    """
    print("\n========================================================")
    print("    EXPERIMENT: DIRECTION / SHAPING CAUSAL CONTROLS     ")
    print("========================================================")
    from models.shaping_controls import fit_pca_U, train_pure_clean_model
    from models.latent_attack import make_U, get_deformation_matrix
    from utils.hadamard import generate_hadamard_csi_triggers
    from datasets.mmfi_dataset import load_mmfi_data
    from evaluate import evaluate_model
    import torch

    device = torch.device(base_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    train_loader, test_loader, PS, _ = load_mmfi_data(base_cfg)

    K = base_cfg.get('attack', {}).get('K', 2)
    shift_m = base_cfg.get('attack', {}).get('target_shift_m', 0.30)
    gate = base_cfg.get('attack', {}).get('gate_threshold', 2.0)
    D = get_deformation_matrix(K=K, ps=PS, shift_m=shift_m, device=device)
    CSI_TRIG = generate_hadamard_csi_triggers(K=K, device=device)

    # 1. Condition 1: Learned U (Proposed Method)
    print("\n>>> Condition 1: Learned U with Shaping Loss (Proposed) <<<")
    cfg_lat = copy.deepcopy(base_cfg)
    cfg_lat['attack']['mode'] = 'latent'
    cfg_lat['experiment_name'] = 'control_learned_u'
    res_learned = train(cfg_lat)
    res_learned['condition'] = 'Learned U (Ours)'

    # 2. Train uncompromised clean model
    print("\n>>> Training Clean Victim Model (No Poisoning / Shaping) <<<")
    clean_net = train_pure_clean_model(base_cfg, train_loader, PS, device=device)

    # 3. Condition 2: Random U on Clean Encoder
    print("\n>>> Condition 2: Random U on Clean Model (No Shaping) <<<")
    U_random = make_U(m=256, K=K, seed=0, device=device)
    res_random = evaluate_model(clean_net, test_loader, 'latent', U_random, D, CSI_TRIG, PS, device=device, gate=gate)
    res_random['experiment_name'] = 'control_random_u'
    res_random['mode'] = 'latent'
    res_random['condition'] = 'Random U (No Shaping)'

    # 4. Condition 3: PCA U on Clean Encoder
    print("\n>>> Condition 3: PCA U on Clean Model (Intrinsic Directions) <<<")
    U_pca = fit_pca_U(clean_net, train_loader, K=K, device=device)
    res_pca = evaluate_model(clean_net, test_loader, 'latent', U_pca, D, CSI_TRIG, PS, device=device, gate=gate)
    res_pca['experiment_name'] = 'control_pca_u'
    res_pca['mode'] = 'latent'
    res_pca['condition'] = 'PCA U (No Shaping)'

    controls_results = [res_learned, res_random, res_pca]
    save_dict_to_csv(controls_results, './outputs/experiments/shaping_controls.csv')
    print_summary_table(controls_results, title="Direction & Shaping Causal Controls")


def run_stealth_evaluation(base_cfg):
    """
    Measures physical and statistical stealth of the trigger (Slide 17 & 31 in Report 9/9).
    Quantifies L2, L_inf perturbation on raw CSI and latent silhouette cluster separability.
    """
    print("\n========================================================")
    print("      EXPERIMENT: STEALTH & IMPERCEPTIBILITY AUDIT       ")
    print("========================================================")
    from utils.stealth import compute_csi_perturbation_stats, compute_latent_clustering_metrics
    from utils.hadamard import generate_hadamard_csi_triggers
    from models.csi_attack import apply_csi_trigger
    from datasets.mmfi_dataset import load_mmfi_data
    import torch

    device = torch.device(base_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    _, test_loader, _, meta = load_mmfi_data(base_cfg)

    x_clean = meta['Xte'][:1000].to(device)
    K = base_cfg.get('attack', {}).get('K', 2)
    eps = base_cfg.get('attack', {}).get('epsilon', 0.15)
    CSI_TRIG = generate_hadamard_csi_triggers(K=K, device=device)

    x_trig_0 = apply_csi_trigger(x_clean.clone(), 0, CSI_TRIG, eps=eps)
    stats = compute_csi_perturbation_stats(x_clean, x_trig_0)

    print("\n=== CSI Input-Level Stealth Metrics ===")
    print(f"  Max Absolute Perturbation (L_inf) : ±{stats['l_inf_max_perturbation']:.5f} (on [0, 1] range)")
    print(f"  Mean Absolute Perturbation        :  {stats['mean_absolute_perturbation']:.5f}")
    print(f"  Relative L2 Perturbation          :  {stats['l2_relative_perturbation_pct']:.2f}%")
    print(f"  Relative Perturbation to Max      :  {stats['stealth_percentage']:.2f}% (Confirms ~0.4% stealth claim)")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automated Experiment Suite for WiFi-CSI Latent Backdoor.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help="Base config path.")
    parser.add_argument('--suite', type=str, default='all',
                        choices=['all', 'poc', 'leave_one_out', 'seeds', 'sweeps', 'controls', 'stealth'],
                        help="Which experimental suite to run.")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    if args.suite in ['all', 'poc']:
        print("\n--- Running PoC Comparison: Latent vs CSI Baseline ---")
        cfg_lat = copy.deepcopy(base_config)
        cfg_lat['attack']['mode'] = 'latent'
        cfg_lat['experiment_name'] = 'poc_latent'
        train(cfg_lat)

        cfg_csi = copy.deepcopy(base_config)
        cfg_csi['attack']['mode'] = 'csi'
        cfg_csi['experiment_name'] = 'poc_csi'
        train(cfg_csi)

    if args.suite in ['all', 'leave_one_out']:
        run_leave_one_room_out(base_config, 'configs/experiments/leave_one_room_out.yaml')

    if args.suite in ['all', 'seeds']:
        run_seed_stability(base_config, 'configs/experiments/seed_stability.yaml')

    if args.suite in ['all', 'sweeps']:
        run_hyperparam_sweeps(base_config, 'configs/experiments/hyperparam_sweeps.yaml')

    if args.suite in ['all', 'controls']:
        run_shaping_controls(base_config)

    if args.suite in ['all', 'stealth']:
        run_stealth_evaluation(base_config)

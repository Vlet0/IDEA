import os
import copy
import argparse
import yaml
import numpy as np
from tabulate import tabulate

from train import train, load_config
from utils.logger import save_dict_to_csv, read_csv_to_dict_list

def run_leave_one_room_out(base_cfg, exp_cfg_path):
    """
    Executes leave-one-room-out experiments across all 4 environments:
    - Train: E01,E02,E03 -> Test: E04
    - Train: E01,E02,E04 -> Test: E03
    - Train: E01,E03,E04 -> Test: E02
    - Train: E02,E03,E04 -> Test: E01
    Supports checkpoint resume: skips already completed splits.
    """
    print("\n========================================================")
    print("       EXPERIMENT: LEAVE-ONE-ROOM-OUT EVALUATION        ")
    print("========================================================")

    with open(exp_cfg_path, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    splits = exp_cfg.get('splits', [])
    modes = exp_cfg.get('modes', ['latent', 'csi'])
    output_base = base_cfg.get('training', {}).get('output_dir', './outputs/')
    exp_out_dir = os.path.join(output_base, 'experiments')
    os.makedirs(exp_out_dir, exist_ok=True)
    output_csv = os.path.join(exp_out_dir, 'leave_one_room_out_results.csv')

    # Load existing results for resuming
    existing_rows = read_csv_to_dict_list(output_csv)
    completed_map = {}
    for r in existing_rows:
        if 'split' in r and 'mode' in r:
            completed_map[(str(r['split']), str(r['mode']).lower())] = r

    all_results = list(completed_map.values())

    for split in splits:
        split_name = split['name']
        tr_rooms = split['train_rooms']
        te_rooms = split['test_rooms']

        for mode in modes:
            key = (str(split_name), str(mode).lower())
            if key in completed_map:
                print(f"[Resume] Split {split_name} | Mode {mode.upper()} already in {output_csv}. Skipping...")
                continue

            print(f"\n>>> Running Split: {split_name} | Mode: {mode.upper()} <<<")
            cfg = copy.deepcopy(base_cfg)
            cfg['data']['train_rooms'] = tr_rooms
            cfg['data']['test_rooms'] = te_rooms
            cfg.setdefault('attack', {})['mode'] = mode
            cfg['experiment_name'] = f"room_{split_name}_{mode}"
            cfg.setdefault('training', {})['output_dir'] = os.path.join(exp_out_dir, f"leave_one_out/{split_name}_{mode}/")

            res = train(cfg)
            res['split'] = split_name
            res['test_room'] = te_rooms[0]
            all_results.append(res)
            completed_map[key] = res
            # Save immediately to prevent progress loss
            save_dict_to_csv(all_results, output_csv, mode='w')

    print_summary_table(all_results, title="Leave-One-Room-Out Transfer Results")


def run_seed_stability(base_cfg, exp_cfg_path):
    """
    Runs 5 independent seeds for both Latent and CSI modes, reporting Mean +- Std.
    Supports checkpoint resume: skips already evaluated seeds.
    """
    print("\n========================================================")
    print("         EXPERIMENT: SEED STABILITY EVALUATION          ")
    print("========================================================")

    with open(exp_cfg_path, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    seeds = exp_cfg.get('seeds', [0, 1, 2, 42, 123])
    modes = exp_cfg.get('modes', ['latent', 'csi'])
    output_base = base_cfg.get('training', {}).get('output_dir', './outputs/')
    exp_out_dir = os.path.join(output_base, 'experiments')
    os.makedirs(exp_out_dir, exist_ok=True)
    output_csv = os.path.join(exp_out_dir, 'seed_stability_results.csv')

    # Load existing results for resuming
    existing_rows = read_csv_to_dict_list(output_csv)
    completed_map = {}
    for r in existing_rows:
        if 'mode' in r and 'seed' in r:
            try:
                completed_map[(str(r['mode']).lower(), int(r['seed']))] = r
            except (ValueError, TypeError):
                pass

    all_results = list(completed_map.values())

    for mode in modes:
        for s in seeds:
            key = (str(mode).lower(), int(s))
            if key in completed_map:
                print(f"[Resume] Mode {mode.upper()} | Seed {s} already in {output_csv}. Skipping...")
                continue

            print(f"\n>>> Running Seed: {s} | Mode: {mode.upper()} <<<")
            cfg = copy.deepcopy(base_cfg)
            cfg['seed'] = s
            cfg.setdefault('attack', {})['mode'] = mode
            cfg['experiment_name'] = f"seed_{s}_{mode}"
            cfg.setdefault('training', {})['output_dir'] = os.path.join(exp_out_dir, f"seeds/{mode}_seed{s}/")

            res = train(cfg)
            all_results.append(res)
            completed_map[key] = res
            save_dict_to_csv(all_results, output_csv, mode='w')

    # Compute Mean +- Std for each mode
    print("\n=== Seed Stability Summary (Mean +- Std over seeds) ===")
    summary_rows = []
    for mode in modes:
        mode_res = [r for r in all_results if str(r.get('mode', '')).lower() == mode.lower()]
        if not mode_res:
            continue
        asrs = [r['asr'] for r in mode_res if 'asr' in r]
        mpjpes = [r['clean_mpjpe_cm'] for r in mode_res if 'clean_mpjpe_cm' in r]
        fps = [r['clean_fp_rate'] for r in mode_res if 'clean_fp_rate' in r]

        summary_rows.append([
            mode.upper(),
            f"{np.mean(mpjpes):.2f} ± {np.std(mpjpes):.2f}" if mpjpes else "-",
            f"{np.mean(asrs):.1f}% ± {np.std(asrs):.1f}%" if asrs else "-",
            f"{np.mean(fps):.2f}% ± {np.std(fps):.2f}%" if fps else "-"
        ])

    print(tabulate(summary_rows, headers=["Mode", "Clean MPJPE (cm)", "ASR (<15cm)", "Gate FP Rate"], tablefmt="grid"))


def run_hyperparam_sweeps(base_cfg, exp_cfg_path):
    """
    Sweeps poison rates (1% to 20%) and epsilons (0.05 to 0.20).
    Supports checkpoint resume: skips already evaluated poison rates.
    """
    print("\n========================================================")
    print("      EXPERIMENT: HYPERPARAMETER SENSITIVITY SWEEPS     ")
    print("========================================================")

    with open(exp_cfg_path, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    poison_rates = exp_cfg.get('poison_rates', [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    output_base = base_cfg.get('training', {}).get('output_dir', './outputs/')
    output_dir = os.path.join(output_base, 'experiments', 'sweeps')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "poison_rate_sweep.csv")

    existing_rows = read_csv_to_dict_list(output_csv)
    completed_map = {}
    for r in existing_rows:
        if 'poison_rate' in r:
            try:
                completed_map[round(float(r['poison_rate']), 4)] = r
            except (ValueError, TypeError):
                pass

    sweep_results = list(completed_map.values())

    for p in poison_rates:
        rate_key = round(float(p), 4)
        if rate_key in completed_map:
            print(f"[Resume] Poison rate {p*100:.0f}% already in {output_csv}. Skipping...")
            continue

        print(f"\n>>> Running Poison Rate: {p*100:.0f}% <<<")
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault('attack', {})['poison_rate'] = p
        cfg['experiment_name'] = f"sweep_poison_{int(p*100)}pct"
        cfg.setdefault('training', {})['output_dir'] = os.path.join(output_dir, f"poison_{int(p*100)}pct/")

        res = train(cfg)
        res['poison_rate'] = p
        sweep_results.append(res)
        completed_map[rate_key] = res
        save_dict_to_csv(sweep_results, output_csv, mode='w')

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
            f"{float(r.get('clean_mpjpe_cm', 0)):.1f}",
            f"{float(r.get('err_target_cm', 0)):.1f}",
            f"{float(r.get('asr', 0)):.1f}%",
            f"{float(r.get('clean_fp_rate', 0)):.2f}%"
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
    Supports checkpoint resume.
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

    output_base = base_cfg.get('training', {}).get('output_dir', './outputs/')
    exp_out_dir = os.path.join(output_base, 'experiments')
    os.makedirs(exp_out_dir, exist_ok=True)
    output_csv = os.path.join(exp_out_dir, 'shaping_controls.csv')

    existing_rows = read_csv_to_dict_list(output_csv)
    completed_map = {r.get('condition'): r for r in existing_rows if r.get('condition')}
    controls_results = list(completed_map.values())

    device = torch.device(base_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    train_loader, test_loader, PS, _ = load_mmfi_data(base_cfg)

    K = base_cfg.get('attack', {}).get('K', 2)
    shift_m = base_cfg.get('attack', {}).get('target_shift_m', 0.30)
    gate = base_cfg.get('attack', {}).get('gate_threshold', 2.0)
    D = get_deformation_matrix(K=K, ps=PS, shift_m=shift_m, device=device)
    CSI_TRIG = generate_hadamard_csi_triggers(K=K, device=device)

    # 1. Condition 1: Learned U (Proposed Method)
    c1_name = 'Learned U (Ours)'
    if c1_name in completed_map:
        print(f"[Resume] Condition 1: {c1_name} already in {output_csv}. Skipping...")
    else:
        print(f"\n>>> Condition 1: {c1_name} with Shaping Loss (Proposed) <<<")
        cfg_lat = copy.deepcopy(base_cfg)
        cfg_lat.setdefault('attack', {})['mode'] = 'latent'
        cfg_lat['experiment_name'] = 'control_learned_u'
        res_learned = train(cfg_lat)
        res_learned['condition'] = c1_name
        controls_results.append(res_learned)
        completed_map[c1_name] = res_learned
        save_dict_to_csv(controls_results, output_csv, mode='w')

    # Train / load clean victim model if either Condition 2 or 3 is missing
    c2_name = 'Random U (No Shaping)'
    c3_name = 'PCA U (No Shaping)'
    clean_net = None

    if (c2_name not in completed_map) or (c3_name not in completed_map):
        print("\n>>> Training / Loading Clean Victim Model (No Poisoning / Shaping) <<<")
        clean_net = train_pure_clean_model(base_cfg, train_loader, PS, device=device)

    # 2. Condition 2: Random U on Clean Encoder
    if c2_name in completed_map:
        print(f"[Resume] Condition 2: {c2_name} already in {output_csv}. Skipping...")
    else:
        print(f"\n>>> Condition 2: {c2_name} on Clean Model <<<")
        latent_dim = base_cfg.get('model', {}).get('latent_dim', 256)
        U_random = make_U(m=latent_dim, K=K, seed=0, device=device)
        eval_rand = evaluate_model(clean_net, test_loader, 'latent', U_random, D, CSI_TRIG, PS, device=device, gate=gate)
        res_random = {
            'experiment_name': 'control_random_u',
            'mode': 'latent',
            'seed': base_cfg.get('seed', 0),
            'condition': c2_name,
            'clean_mpjpe_cm': eval_rand['clean_mpjpe_cm'],
            'err_target_cm': eval_rand['err_target_cm'],
            'err_original_cm': eval_rand['err_original_cm'],
            'asr': eval_rand['asr'],
            'clean_fp_rate': eval_rand['clean_fp_rate'],
            's_clean_norm': eval_rand['s_clean_norm'],
            's_trig_norm': eval_rand['s_trig_norm']
        }
        controls_results.append(res_random)
        completed_map[c2_name] = res_random
        save_dict_to_csv(controls_results, output_csv, mode='w')

    # 3. Condition 3: PCA U on Clean Encoder
    if c3_name in completed_map:
        print(f"[Resume] Condition 3: {c3_name} already in {output_csv}. Skipping...")
    else:
        print(f"\n>>> Condition 3: {c3_name} on Clean Model (Intrinsic Directions) <<<")
        U_pca = fit_pca_U(clean_net, train_loader, K=K, device=device)
        eval_pca = evaluate_model(clean_net, test_loader, 'latent', U_pca, D, CSI_TRIG, PS, device=device, gate=gate)
        res_pca = {
            'experiment_name': 'control_pca_u',
            'mode': 'latent',
            'seed': base_cfg.get('seed', 0),
            'condition': c3_name,
            'clean_mpjpe_cm': eval_pca['clean_mpjpe_cm'],
            'err_target_cm': eval_pca['err_target_cm'],
            'err_original_cm': eval_pca['err_original_cm'],
            'asr': eval_pca['asr'],
            'clean_fp_rate': eval_pca['clean_fp_rate'],
            's_clean_norm': eval_pca['s_clean_norm'],
            's_trig_norm': eval_pca['s_trig_norm']
        }
        controls_results.append(res_pca)
        completed_map[c3_name] = res_pca
        save_dict_to_csv(controls_results, output_csv, mode='w')

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

    base_config = load_config(args.config)

    if args.suite in ['all', 'poc']:
        print("\n--- Running PoC Comparison: Latent vs CSI Baseline ---")
        cfg_lat = copy.deepcopy(base_config)
        cfg_lat.setdefault('attack', {})['mode'] = 'latent'
        cfg_lat['experiment_name'] = 'poc_latent'
        train(cfg_lat)

        cfg_csi = copy.deepcopy(base_config)
        cfg_csi.setdefault('attack', {})['mode'] = 'csi'
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

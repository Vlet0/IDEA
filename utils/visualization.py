import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datasets.transforms import BONES

def plot_skeleton_demo(clean_pose, trig_poses, trig_scores, output_path="./outputs/skeleton_demo.png"):
    """
    Renders 3D skeleton comparison matching cell 37:
    Panels: Clean reference vs Triggered poses with latent score annotation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_panels = 1 + len(trig_poses)
    fig = plt.figure(figsize=(4.5 * num_panels, 4.2))
    
    # 1. Clean panel
    ax0 = fig.add_subplot(1, num_panels, 1, projection='3d')
    P = clean_pose
    for a, b in BONES:
        ax0.plot([P[a, 0], P[b, 0]], [P[a, 2], P[b, 2]], [-P[a, 1], -P[b, 1]],
                 c='#1F4E79', lw=2.2)
    ax0.scatter(P[:, 0], P[:, 2], -P[:, 1], c='k', s=9)
    ax0.set_title("Clean input\n(Reference pose)", fontsize=10)
    ax0.set_xlim(-1, 1); ax0.set_ylim(-1, 1); ax0.set_zlim(-1, 1)
    ax0.set_xticklabels([]); ax0.set_yticklabels([]); ax0.set_zticklabels([])

    # 2. Trigger panels
    for idx, (P_trig, s) in enumerate(zip(trig_poses, trig_scores)):
        ax = fig.add_subplot(1, num_panels, idx + 2, projection='3d')
        # Draw faint clean skeleton as reference
        for a, b in BONES:
            ax.plot([P[a, 0], P[b, 0]], [P[a, 2], P[b, 2]], [-P[a, 1], -P[b, 1]],
                    c='#B0C4DE', lw=1.2, linestyle=':')
        # Draw deformed triggered skeleton
        for a, b in BONES:
            ax.plot([P_trig[a, 0], P_trig[b, 0]], [P_trig[a, 2], P_trig[b, 2]], [-P_trig[a, 1], -P_trig[b, 1]],
                    c='#B23A2E', lw=2.2)
        ax.scatter(P_trig[:, 0], P_trig[:, 2], -P_trig[:, 1], c='k', s=9)
        score_str = ", ".join([f"{val:+.2f}" for val in s[:2]])
        ax.set_title(f"Trigger k={idx}\ns=[{score_str}]", fontsize=10)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved 3D skeleton figure to {output_path}")

def plot_latent_distribution(s_clean, s_trig, gate=2.0, output_path="./outputs/latent_distribution.png"):
    """
    Renders histogram of max |s_k| for clean vs triggered frames matching Slide 15.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sc = np.max(np.abs(s_clean), axis=-1)
    st = np.max(np.abs(s_trig), axis=-1)
    
    plt.figure(figsize=(7, 4.5))
    plt.hist(sc, bins=50, alpha=0.7, color='#1F4E79', label='Clean', density=False)
    plt.hist(st, bins=50, alpha=0.7, color='#B23A2E', label='Triggered', density=False)
    plt.axvline(x=gate, color='black', linestyle='--', linewidth=1.5, label=f'Gate={gate}')
    
    plt.title("Clean vs Triggered Latent Separation", fontsize=12)
    plt.xlabel(r"$\max_k |s_k|$ (Trigger strength)", fontsize=11)
    plt.ylabel("# Frames", fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[Visualization] Saved latent distribution to {output_path}")

def plot_gate_sweep(gates, asrs, fprs, mpjpes, output_path="./outputs/gate_sweep.png"):
    """
    Plots ASR, FPR and Clean MPJPE vs Gate Threshold matching Slide 17.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(gates, asrs, 'o-', color='#B23A2E', linewidth=2, label='ASR (%)')
    ax1.plot(gates, fprs, 's-', color='#1F4E79', linewidth=2, label='Gate False Positive (%)')
    ax1.set_xlabel('Gate Threshold', fontsize=11)
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    ax2.plot(gates, mpjpes, '^--', color='gray', linewidth=1.5, label='Clean MPJPE (cm)')
    ax2.set_ylabel('Clean MPJPE (cm)', color='gray', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='gray')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title("Stealth: Activation Threshold Operating Region", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[Visualization] Saved gate sweep plot to {output_path}")

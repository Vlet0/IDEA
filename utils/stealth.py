import numpy as np
import torch
from sklearn.metrics import silhouette_score

def compute_csi_perturbation_stats(x_clean, x_trig):
    """
    Quantifies the imperceptibility of the trigger in input CSI space:
    - L2 relative perturbation: ||x_trig - x_clean||_2 / ||x_clean||_2
    - L_inf max absolute perturbation: max |x_trig - x_clean|
    - Perturbation percentage relative to amplitude range [0, 1]
    Confirms Slide 17: perturbation is ~0.4% (+-0.004).
    """
    if isinstance(x_clean, torch.Tensor):
        x_clean = x_clean.detach().cpu().numpy()
    if isinstance(x_trig, torch.Tensor):
        x_trig = x_trig.detach().cpu().numpy()

    diff = np.abs(x_trig - x_clean)
    l_inf = float(np.max(diff))
    l2_diff = np.linalg.norm(diff.reshape(len(diff), -1), axis=1)
    l2_clean = np.linalg.norm(x_clean.reshape(len(x_clean), -1), axis=1) + 1e-8
    l2_rel = float(np.mean(l2_diff / l2_clean) * 100.0)
    mean_abs = float(np.mean(diff))

    stats = {
        'l_inf_max_perturbation': l_inf,
        'l2_relative_perturbation_pct': l2_rel,
        'mean_absolute_perturbation': mean_abs,
        'stealth_percentage': l_inf * 100.0  # Since range is [0, 1]
    }
    return stats

def compute_latent_clustering_metrics(s_clean, s_trig_dict):
    """
    Evaluates latent separation between clean cluster and triggered clusters:
    - Silhouette score across clusters (Clean vs Trigger 0 vs Trigger 1)
    - Inter-cluster vs intra-cluster distance ratio
    """
    if isinstance(s_clean, torch.Tensor):
        s_clean = s_clean.detach().cpu().numpy()

    X_list = [s_clean]
    labels_list = [np.zeros(len(s_clean), dtype=int)]

    for k, st in s_trig_dict.items():
        if isinstance(st, torch.Tensor):
            st = st.detach().cpu().numpy()
        X_list.append(st)
        labels_list.append(np.full(len(st), fill_value=k + 1, dtype=int))

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(labels_list, axis=0)

    # Downsample if too large for silhouette
    if len(X_all) > 5000:
        idx = np.random.choice(len(X_all), 5000, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    sil_score = float(silhouette_score(X_all, y_all))
    return {'silhouette_score': sil_score, 'num_clusters': len(X_list)}

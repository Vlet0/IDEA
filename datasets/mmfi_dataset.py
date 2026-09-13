import os
import glob
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import normalize_pose

class MMFIDataset(Dataset):
    """
    PyTorch Dataset for MM-Fi WiFi-CSI Pose Estimation.
    Takes pre-normalized tensors X and Y.
    """
    def __init__(self, X, Y, env=None, subjects=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        self.X = X
        self.Y = Y
        self.env = env
        self.subjects = subjects

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_subject_scene(subject_str):
    """Maps subject identifier to environment scene."""
    try:
        sub_num = int(subject_str.replace('S', ''))
        if 1 <= sub_num <= 10:
            return 'E01'
        elif 11 <= sub_num <= 20:
            return 'E02'
        elif 21 <= sub_num <= 30:
            return 'E03'
        elif 31 <= sub_num <= 40:
            return 'E04'
    except Exception:
        pass
    return None


def read_csi_mat_frame(mat_path):
    """
    Reads a single .mat frame for wifi-csi modality and applies MM-Fi per-frame min-max normalization.
    Returns (3, 114, 10) array with values in [0, 1].
    """
    data = scio.loadmat(mat_path)['CSIamp']
    data[np.isinf(data)] = np.nan
    # Handle NaN values column by column across the 10 packets
    for i in range(10):
        temp_col = data[:, :, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            mean_val = temp_not_nan_col.mean() if len(temp_not_nan_col) > 0 else 0.0
            temp_col[np.isnan(temp_col)] = mean_val
            
    mn = np.min(data)
    mx = np.max(data)
    if mx - mn > 1e-8:
        data = (data - mn) / (mx - mn)
    else:
        data = np.zeros_like(data)
    return data.astype(np.float32)


def load_mmfi_data(config):
    """
    Unified loader for MM-Fi data according to config:
    - If data.mode == 'cache': loads directly from npz cache file.
    - If data.mode == 'raw': scans raw_root, loads frames, or builds cache on the fly.
    
    Returns:
        train_loader, test_loader, PS (pose scale in meters), dataset_dict
    """
    data_cfg = config.get('data', {})
    mode = data_cfg.get('mode', 'cache')
    cache_path = data_cfg.get('cache_path', './data/mmfi_cache.npz')
    train_rooms = data_cfg.get('train_rooms', ['E01', 'E02', 'E03'])
    test_rooms = data_cfg.get('test_rooms', ['E04'])
    root_j = data_cfg.get('root_joint_idx', 0)

    # 1. Attempt loading from cache first
    if os.path.exists(cache_path):
        print(f"[DataLoader] Loading preprocessed cache from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        X, Y, env = d["X"], d["Y"], d["env"]
    elif mode == 'raw':
        raw_root = data_cfg.get('raw_root', '/media/jackson/Data/wificsi/MMFI/Compress/')
        print(f"[DataLoader] Cache not found at {cache_path}. Building from raw root: {raw_root}...")
        from .prepare_cache import extract_mmfi_cache
        X, Y, env = extract_mmfi_cache(
            raw_root=raw_root,
            actions=data_cfg.get('actions', ['A01', 'A02', 'A03', 'A04']),
            output_cache_path=cache_path
        )
    else:
        raise FileNotFoundError(
            f"Cache file not found at '{cache_path}'. "
            f"Please set data.mode='raw' with raw_root, or run python -m datasets.prepare_cache"
        )

    # 2. Filter train and test splits based on environment
    tr_mask = np.isin(env, train_rooms)
    te_mask = np.isin(env, test_rooms)
    
    print(f"[DataLoader] Train samples ({train_rooms}): {tr_mask.sum()} | Test samples ({test_rooms}): {te_mask.sum()}")
    assert tr_mask.sum() > 0, f"No training samples found for rooms {train_rooms}!"
    assert te_mask.sum() > 0, f"No test samples found for rooms {test_rooms}!"

    # 3. Exact notebook normalization logic:
    # Yc = Y - Y[:, ROOT_J:ROOT_J+1, :]
    # PS  = float(Yc[tr].std())
    Y_norm, PS = normalize_pose(Y, tr_mask=tr_mask, root_j=root_j)
    print(f"[DataLoader] Pose scale PS: {PS:.4f} m")

    Xtr = torch.tensor(X[tr_mask], dtype=torch.float32)
    Ytr = torch.tensor(Y_norm[tr_mask], dtype=torch.float32)
    Xte = torch.tensor(X[te_mask], dtype=torch.float32)
    Yte = torch.tensor(Y_norm[te_mask], dtype=torch.float32)

    print(f"[DataLoader] Xtr: {Xtr.shape}, Ytr: {Ytr.shape}")
    print(f"[DataLoader] Xte: {Xte.shape}, Yte: {Yte.shape}")

    train_dataset = MMFIDataset(Xtr, Ytr, env=env[tr_mask])
    test_dataset = MMFIDataset(Xte, Yte, env=env[te_mask])

    batch_size = config.get('training', {}).get('batch_size', 256)
    eval_bs = config.get('evaluation', {}).get('batch_size', 512)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False, drop_last=False)

    meta = {
        'PS': PS,
        'Xtr': Xtr,
        'Ytr': Ytr,
        'Xte': Xte,
        'Yte': Yte,
        'tr_mask': tr_mask,
        'te_mask': te_mask
    }

    return train_loader, test_loader, PS, meta

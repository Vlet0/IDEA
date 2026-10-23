import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .prepare_cache import SCENE_SUBJECTS, SUBJECTS_PER_ROOM
from .transforms import canonicalize_pose_z_up, normalize_pose

class MMFIDataset(Dataset):
    """
    PyTorch Dataset for MM-Fi WiFi-CSI Pose Estimation.
    Takes pre-normalized tensors X and Y.
    """
    def __init__(self, X, Y, env=None, subjects=None, is_fall=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        self.X = X
        self.Y = Y
        self.env = env
        self.subjects = subjects
        self.is_fall = None
        if is_fall is not None:
            labels = torch.as_tensor(is_fall)
            if labels.shape != (len(X),) or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("is_fall must be a binary vector with one label per frame")
            self.is_fall = labels.bool()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_fall is not None:
            return self.X[idx], self.Y[idx], self.is_fall[idx]
        return self.X[idx], self.Y[idx]


def load_mmfi_data(config):
    """
    Unified loader for MM-Fi data.  ``config`` must be a resolved run config
    (``utils.config.resolve_config``) so every key below is present.

    - ``data.mode == 'cache'``: load the preprocessed npz cache (built on the fly from
      ``raw_root`` when the cache file is missing and ``mode == 'raw'``).

    Returns:
        train_loader, test_loader, PS (pose scale in metres), meta dict.
    """
    data_cfg = config['data']
    mode = data_cfg['mode']
    cache_path = data_cfg['cache_path']
    train_rooms = data_cfg['train_rooms']
    test_rooms = data_cfg['test_rooms']
    root_j = data_cfg['root_joint_idx']
    input_shape = tuple(data_cfg['input_shape'])
    num_joints, joint_dim = data_cfg['num_joints'], data_cfg['joint_dim']
    fall_labels = None

    if os.path.exists(cache_path):
        print(f"[DataLoader] Loading preprocessed cache from {cache_path}")
        with np.load(cache_path, allow_pickle=False) as d:
            required_keys = {'X', 'Y', 'env'}
            missing_keys = required_keys.difference(d.files)
            if missing_keys:
                raise ValueError(f"Invalid MM-Fi cache; missing arrays: {sorted(missing_keys)}")
            X, Y, env = d["X"], d["Y"], d["env"]
            if "is_fall" in d.files:
                fall_labels = d["is_fall"]
                print(f"[DataLoader Debug] Found 'is_fall' key in cache: shape={fall_labels.shape}, falls={int(fall_labels.sum())}")
            else:
                # Check for separate fall labels file alongside cache
                cache_dir = os.path.dirname(os.path.abspath(cache_path))
                candidates = [
                    cache_path.replace('.npz', '_is_fall.npy'),
                    os.path.join(cache_dir, "is_fall.npy"),
                    os.path.join(cache_dir, "is_fall_labels.npy"),
                    data_cfg["fall_labels_path"],
                ]
                for cand in candidates:
                    if cand and os.path.exists(cand):
                        try:
                            fall_labels = np.load(cand)
                            print(f"[DataLoader Debug] Loaded external is_fall labels from: {cand}")
                            break
                        except Exception as e:
                            print(f"[DataLoader Warning] Failed to load {cand}: {e}")
    elif mode == 'raw':
        raw_root = data_cfg['raw_root']
        print(f"[DataLoader] Cache not found at {cache_path}. Building from raw root: {raw_root}...")
        from .prepare_cache import extract_mmfi_cache
        X, Y, env = extract_mmfi_cache(
            raw_root=raw_root,
            actions=data_cfg['actions'],
            output_cache_path=cache_path,
            max_frames_per_action=data_cfg['expected_frames_per_action'],
        )
    else:
        raise FileNotFoundError(
            f"Cache file not found at '{cache_path}'. "
            f"Please set data.mode='raw' with raw_root, or run python -m datasets.prepare_cache"
        )

    if X.ndim != 4 or tuple(X.shape[1:]) != input_shape:
        raise ValueError(f"Invalid MM-Fi CSI shape in cache: {X.shape}; expected (N, *{input_shape})")
    if Y.ndim != 3 or tuple(Y.shape[1:]) != (num_joints, joint_dim):
        raise ValueError(f"Invalid MM-Fi pose shape in cache: {Y.shape}; expected (N, {num_joints}, {joint_dim})")
    if len(X) != len(Y) or len(X) != len(env):
        raise ValueError(f"MM-Fi cache arrays have different lengths: X={len(X)}, Y={len(Y)}, env={len(env)}")
    if fall_labels is not None:
        if fall_labels.shape != (len(X),) or not np.isin(fall_labels, (0, 1)).all():
            raise ValueError("Cache is_fall must be a binary vector of length N")
        n_fall = int(fall_labels.sum())
        print(f"[DataLoader Debug] Using annotated is_fall labels: {n_fall}/{len(fall_labels)} frames are fall.")
    else:
        print("[DataLoader Debug] No is_fall annotations found in cache or external files;")
        print("[DataLoader Debug] Evaluation will derive fall labels from ground-truth pose geometry.")

    expected_frames = data_cfg['expected_frames_per_action']
    if data_cfg['require_complete_cache'] and expected_frames is not None:
        expected_per_room = SUBJECTS_PER_ROOM * len(data_cfg['actions']) * int(expected_frames)
        incomplete_rooms = {
            room: int(np.count_nonzero(env == room))
            for room in SCENE_SUBJECTS
            if int(np.count_nonzero(env == room)) != expected_per_room
        }
        if incomplete_rooms:
            raise ValueError(
                f"Incomplete MM-Fi cache: expected {expected_per_room} samples per room, "
                f"found {incomplete_rooms}. Rebuild it from a complete raw dataset."
            )

    # 2. Filter train and test splits based on environment
    tr_mask = np.isin(env, train_rooms)
    te_mask = np.isin(env, test_rooms)
    
    print(f"[DataLoader] Train samples ({train_rooms}): {tr_mask.sum()} | Test samples ({test_rooms}): {te_mask.sum()}")
    assert tr_mask.sum() > 0, f"No training samples found for rooms {train_rooms}!"
    assert te_mask.sum() > 0, f"No test samples found for rooms {test_rooms}!"

    # 3. Canonical Z-up coordinates, then exact root-centred normalization.
    # Yc = Y - Y[:, ROOT_J:ROOT_J+1, :]
    # PS  = float(Yc[tr].std())
    Y_norm, PS = normalize_pose(canonicalize_pose_z_up(Y), tr_mask=tr_mask, root_j=root_j)
    print(f"[DataLoader] Pose scale PS: {PS:.4f} m")

    # Boolean indexing already creates an owned NumPy array.  from_numpy
    # avoids making a second full copy, which matters for the ~47k-frame cache.
    Xtr = torch.from_numpy(np.ascontiguousarray(X[tr_mask], dtype=np.float32))
    Ytr = torch.from_numpy(np.ascontiguousarray(Y_norm[tr_mask], dtype=np.float32))
    Xte = torch.from_numpy(np.ascontiguousarray(X[te_mask], dtype=np.float32))
    Yte = torch.from_numpy(np.ascontiguousarray(Y_norm[te_mask], dtype=np.float32))

    import logging as _logging
    _dl_logger = _logging.getLogger(__name__)

    _dl_logger.info(f"[DataLoader] Split sizes — train: {Xtr.shape[0]} samples, test: {Xte.shape[0]} samples")
    _dl_logger.info(f"[DataLoader] Input shape — Xtr: {tuple(Xtr.shape)}, Xte: {tuple(Xte.shape)}")

    train_dataset = MMFIDataset(Xtr, Ytr, env=env[tr_mask])
    test_is_fall = fall_labels[te_mask] if fall_labels is not None else None
    if test_is_fall is not None:
        n_falls = int(test_is_fall.sum())
        _dl_logger.info(
            f"[DataLoader] Test fall labels: {n_falls}/{len(test_is_fall)} annotated falls "
            f"({100.0 * n_falls / max(len(test_is_fall), 1):.1f}%)."
        )
    else:
        _dl_logger.warning(
            "[DataLoader] Test split has NO annotated fall labels (test_is_fall is None). "
            "Ground-truth fall status will be derived from pose geometry (the alarm definition itself)."
        )

    test_dataset = MMFIDataset(
        Xte, Yte, env=env[te_mask],
        is_fall=test_is_fall,
    )

    batch_size    = config['training']['batch_size']
    eval_bs       = config['evaluation']['batch_size']
    train_workers = int(config['training']['num_workers'])
    eval_workers  = int(config['evaluation']['num_workers'])
    train_pin_memory = bool(config['training']['pin_memory'])
    eval_pin_memory  = bool(config['evaluation']['pin_memory'])
    if train_workers < 0 or eval_workers < 0:
        raise ValueError("DataLoader num_workers must be non-negative")

    _dl_logger.info(
        f"[DataLoader] Train loader — batch_size={batch_size}, num_workers={train_workers}, "
        f"pin_memory={train_pin_memory}, persistent_workers={train_workers > 0}"
    )
    _dl_logger.info(
        f"[DataLoader] Eval  loader — batch_size={eval_bs}, num_workers={eval_workers}, "
        f"pin_memory={eval_pin_memory}, persistent_workers={eval_workers > 0}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=train_workers, persistent_workers=train_workers > 0,
        pin_memory=train_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_bs, shuffle=False, drop_last=False,
        num_workers=eval_workers, persistent_workers=eval_workers > 0,
        pin_memory=eval_pin_memory,
    )

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

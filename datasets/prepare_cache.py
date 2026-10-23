import os
import argparse
import glob
import numpy as np
import scipy.io as scio
from tqdm import tqdm


SUBJECTS_PER_ROOM = 10
SCENE_SUBJECTS = {
    f'E{room:02d}': [f'S{i:02d}' for i in range((room - 1) * SUBJECTS_PER_ROOM + 1, room * SUBJECTS_PER_ROOM + 1)]
    for room in range(1, 5)
}


def validate_mmfi_raw(raw_root, actions, expected_frames_per_action=297, strict=True):
    """Validate the MM-Fi subset before spending time building a cache.

    A partially copied or concurrently modified dataset used to be silently
    accepted, producing a cache with missing rooms/subjects.  The experiment
    split assumes ten subjects per room and a complete set of requested
    actions, so fail early unless the caller explicitly opts out.
    """
    raw_root = os.path.abspath(raw_root)
    actions = tuple(actions)
    errors = []
    total_frames = 0

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"MM-Fi raw root does not exist: {raw_root}")

    for scene, subjects in SCENE_SUBJECTS.items():
        scene_path = os.path.join(raw_root, scene)
        if not os.path.isdir(scene_path):
            errors.append(f"missing scene directory: {scene_path}")
            continue

        for subject in subjects:
            subject_path = os.path.join(scene_path, subject)
            if not os.path.isdir(subject_path):
                errors.append(f"missing subject: {scene}/{subject}")
                continue

            for action in actions:
                action_path = os.path.join(subject_path, action)
                gt_path = os.path.join(action_path, 'ground_truth.npy')
                csi_dir = os.path.join(action_path, 'wifi-csi')

                if not os.path.isfile(gt_path):
                    errors.append(f"missing ground truth: {scene}/{subject}/{action}")
                    continue
                if not os.path.isdir(csi_dir):
                    errors.append(f"missing CSI directory: {scene}/{subject}/{action}")
                    continue

                mat_files = glob.glob(os.path.join(csi_dir, 'frame*.mat'))
                npy_files = glob.glob(os.path.join(csi_dir, '*_processed.npy')) or glob.glob(os.path.join(csi_dir, 'frame*.npy'))
                frame_count = len(mat_files) if mat_files else len(npy_files)
                if frame_count == 0:
                    for single_name in ['csi_processed.npy', 'wifi_csi_processed.npy', 'wifi-csi.npy', 'csi.npy']:
                        single_p = os.path.join(csi_dir, single_name)
                        if not os.path.isfile(single_p):
                            single_p = os.path.join(action_path, single_name)
                        if os.path.isfile(single_p):
                            try:
                                frame_count = int(np.load(single_p, mmap_mode='r').shape[0])
                                break
                            except Exception:
                                pass
                try:
                    gt_count = int(np.load(gt_path, mmap_mode='r').shape[0])
                except Exception as exc:
                    errors.append(f"unreadable ground truth: {scene}/{subject}/{action}: {exc}")
                    continue

                usable = min(frame_count, gt_count, expected_frames_per_action)
                total_frames += usable
                if frame_count < expected_frames_per_action or gt_count < expected_frames_per_action:
                    errors.append(
                        f"incomplete action: {scene}/{subject}/{action} "
                        f"(CSI={frame_count}, ground_truth={gt_count}, "
                        f"expected>={expected_frames_per_action})"
                    )

    expected_actions = sum(len(v) for v in SCENE_SUBJECTS.values()) * len(actions)
    expected_total = expected_actions * expected_frames_per_action

    if errors:
        preview = '\n'.join(f"  - {item}" for item in errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more problem(s)" if len(errors) > 20 else ''
        message = (
            f"MM-Fi raw-data validation failed with {len(errors)} problem(s):\n"
            f"{preview}{suffix}\n"
            "Restore/finish copying the dataset before creating a cache. "
            "Use --allow-incomplete only for an intentional debug subset."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[WARNING] {message}")
    else:
        print(
            f"Raw-data validation passed: {expected_actions} subject/action pairs, "
            f"{expected_total} frames."
        )

    return {
        'errors': errors,
        'usable_frames': total_frames,
        'expected_frames': expected_total,
    }

def read_csi_mat(mat_path):
    """Loads CSI amplitude from .mat and normalizes to [0, 1]."""
    try:
        data = scio.loadmat(mat_path)['CSIamp']
        data[np.isinf(data)] = np.nan
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
    except Exception as e:
        print(f"Error reading {mat_path}: {e}")
        return None

def read_csi_file(file_path):
    """Loads CSI frame from either .mat or .npy format with normalization."""
    if file_path.endswith('.mat'):
        return read_csi_mat(file_path)
    try:
        data = np.load(file_path)
        if data.shape == (10, 114, 3):
            data = np.transpose(data, (2, 1, 0))
        elif data.shape == (114, 10, 3):
            data = np.transpose(data, (2, 0, 1))
        elif data.shape != (3, 114, 10) and data.size == 3 * 114 * 10:
            data = data.reshape(3, 114, 10)
        
        data[np.isinf(data)] = np.nan
        for i in range(data.shape[2] if data.ndim == 3 else 10):
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
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

ALL_MMFI_ACTIONS = [f'A{i:02d}' for i in range(1, 28)]


def extract_mmfi_cache(
    raw_root='../Compress/',
    actions=None,
    output_cache_path='./data/mmfi_cache_poc.npz',
    max_frames_per_action=297,
    require_complete=True,
):
    if actions is None or actions == 'all':
        actions = ALL_MMFI_ACTIONS

    """
    Scans raw MMFI directory:
      raw_root/{scene}/{subject}/{action}/wifi-csi/frame{idx}.mat
      raw_root/{scene}/{subject}/{action}/ground_truth.npy
    Compacts them into a single .npz cache:
      X: (N, 3, 114, 10)
      Y: (N, 17, 3)
      env: (N,)
    """
    print(f"=== Scanning MMFI dataset at: {raw_root} ===")
    validate_mmfi_raw(
        raw_root,
        actions,
        expected_frames_per_action=max_frames_per_action,
        strict=require_complete,
    )

    all_scenes = list(SCENE_SUBJECTS)
    all_subjects = [s for subjects in SCENE_SUBJECTS.values() for s in subjects]
    subject_room = {s: room for room, subjects in SCENE_SUBJECTS.items() for s in subjects}

    X_list = []
    Y_list = []
    env_list = []

    total_frames = 0
    for scene in all_scenes:
        scene_path = os.path.join(raw_root, scene)
        if not os.path.isdir(scene_path):
            # Check if subjects are directly in raw_root
            scene_path = raw_root

        for sub in tqdm(all_subjects, desc=f"Processing {scene}"):
            expected_scene = subject_room[sub]
            if expected_scene != scene and os.path.isdir(os.path.join(raw_root, scene)):
                continue

            sub_path = os.path.join(scene_path, sub) if os.path.isdir(os.path.join(scene_path, sub)) else os.path.join(raw_root, sub)
            if not os.path.isdir(sub_path):
                continue

            for act in actions:
                act_path = os.path.join(sub_path, act)
                if not os.path.isdir(act_path):
                    continue

                gt_path = os.path.join(act_path, 'ground_truth.npy')
                csi_dir = os.path.join(act_path, 'wifi-csi')
                if not os.path.exists(gt_path) or not os.path.isdir(csi_dir):
                    continue

                gt_data = np.load(gt_path)  # (F, 17, 3)
                mat_files = sorted(glob.glob(os.path.join(csi_dir, "frame*.mat")))
                npy_files = sorted(glob.glob(os.path.join(csi_dir, "*_processed.npy")) or glob.glob(os.path.join(csi_dir, "frame*.npy")))
                csi_files = mat_files if mat_files else npy_files

                if csi_files:
                    num_frames = min(len(csi_files), len(gt_data), max_frames_per_action)
                    for f_idx in range(num_frames):
                        csi_frame = read_csi_file(csi_files[f_idx])
                        if csi_frame is not None:
                            X_list.append(csi_frame)
                            Y_list.append(gt_data[f_idx])
                            env_list.append(expected_scene)
                            total_frames += 1
                else:
                    # Check for single sequence array
                    seq_arr = None
                    for single_name in ['csi_processed.npy', 'wifi_csi_processed.npy', 'wifi-csi.npy', 'csi.npy']:
                        single_p = os.path.join(csi_dir, single_name)
                        if not os.path.isfile(single_p):
                            single_p = os.path.join(act_path, single_name)
                        if os.path.isfile(single_p):
                            try:
                                seq_arr = np.load(single_p)
                                break
                            except Exception:
                                pass
                    if seq_arr is not None:
                        if seq_arr.ndim == 4 and seq_arr.shape[1:] == (10, 114, 3):
                            seq_arr = np.transpose(seq_arr, (0, 3, 2, 1))
                        num_frames = min(len(seq_arr), len(gt_data), max_frames_per_action)
                        for f_idx in range(num_frames):
                            frame_data = seq_arr[f_idx].astype(np.float32)
                            # Normalize
                            mn = np.min(frame_data)
                            mx = np.max(frame_data)
                            if mx - mn > 1e-8:
                                frame_data = (frame_data - mn) / (mx - mn)
                            X_list.append(frame_data)
                            Y_list.append(gt_data[f_idx])
                            env_list.append(expected_scene)
                            total_frames += 1

    print(f"Total collected valid frames: {total_frames}")
    if not X_list:
        raise RuntimeError(f"No valid MM-Fi frames were found under: {raw_root}")

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    env = np.array(env_list)

    print(f"Dataset shapes: X={X.shape}, Y={Y.shape}, env={env.shape}")
    output_cache_path = os.path.abspath(output_cache_path)
    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)
    temp_cache_path = f"{output_cache_path}.tmp.{os.getpid()}.npz"
    try:
        np.savez_compressed(temp_cache_path, X=X, Y=Y, env=env)
        os.replace(temp_cache_path, output_cache_path)
    finally:
        if os.path.exists(temp_cache_path):
            os.remove(temp_cache_path)
    print(f"Successfully cached dataset to: {output_cache_path}")
    return X, Y, env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract MM-Fi dataset to fast .npz cache.")
    parser.add_argument('--raw_root', type=str, default='../Compress/',
                        help="Path to raw MMFI Compress directory.")
    parser.add_argument('--output', type=str, default='./data/mmfi_cache_poc.npz',
                        help="Path to output .npz cache file.")
    parser.add_argument('--actions', nargs='+', default=ALL_MMFI_ACTIONS,
                        help="Actions to include (default: all 27 MM-Fi actions A01-A27 matching DT-Pose).")
    parser.add_argument('--expected_frames', type=int, default=297,
                        help="Required frames per subject/action (default: 297).")
    parser.add_argument('--allow-incomplete', action='store_true',
                        help="Allow an intentional debug subset instead of requiring all subjects/actions.")
    parser.add_argument('--validate-only', action='store_true',
                        help="Validate the raw dataset without creating a cache.")
    args = parser.parse_args()
    actions_to_use = ALL_MMFI_ACTIONS if args.actions == ['all'] or args.actions == 'all' else args.actions
    if args.validate_only:
        validate_mmfi_raw(
            args.raw_root,
            actions_to_use,
            expected_frames_per_action=args.expected_frames,
            strict=not args.allow_incomplete,
        )
    else:
        extract_mmfi_cache(
            raw_root=args.raw_root,
            actions=actions_to_use,
            output_cache_path=args.output,
            max_frames_per_action=args.expected_frames,
            require_complete=not args.allow_incomplete,
        )


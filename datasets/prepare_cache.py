import os
import argparse
import glob
import numpy as np
import scipy.io as scio
from tqdm import tqdm

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

def extract_mmfi_cache(
    raw_root='/media/jackson/Data/wificsi/MMFI/Compress/',
    actions=('A01', 'A02', 'A03', 'A04'),
    output_cache_path='./data/mmfi_cache.npz',
    max_frames_per_action=297
):
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
    assert os.path.isdir(raw_root), f"Directory {raw_root} does not exist!"

    all_scenes = ['E01', 'E02', 'E03', 'E04']
    all_subjects = [f"S{i:02d}" for i in range(1, 41)]

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
            # Map subject to room
            sub_id = int(sub[1:])
            expected_scene = (
                'E01' if 1 <= sub_id <= 10 else
                'E02' if 11 <= sub_id <= 20 else
                'E03' if 21 <= sub_id <= 30 else 'E04'
            )
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
                
                num_frames = min(len(mat_files), len(gt_data), max_frames_per_action)
                for f_idx in range(num_frames):
                    csi_frame = read_csi_mat(mat_files[f_idx])
                    if csi_frame is not None:
                        X_list.append(csi_frame)
                        Y_list.append(gt_data[f_idx])
                        env_list.append(expected_scene)
                        total_frames += 1

    print(f"Total collected valid frames: {total_frames}")
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    env = np.array(env_list)

    print(f"Dataset shapes: X={X.shape}, Y={Y.shape}, env={env.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(output_cache_path)), exist_ok=True)
    np.savez_compressed(output_cache_path, X=X, Y=Y, env=env)
    print(f"Successfully cached dataset to: {output_cache_path}")
    return X, Y, env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract MM-Fi dataset to fast .npz cache.")
    parser.add_argument('--raw_root', type=str, default='/media/jackson/Data/wificsi/MMFI/Compress/',
                        help="Path to raw MMFI Compress directory.")
    parser.add_argument('--output', type=str, default='./data/mmfi_cache.npz',
                        help="Path to output .npz cache file.")
    parser.add_argument('--actions', nargs='+', default=['A01', 'A02', 'A03', 'A04'],
                        help="Actions to include.")
    args = parser.parse_args()
    extract_mmfi_cache(raw_root=args.raw_root, actions=args.actions, output_cache_path=args.output)

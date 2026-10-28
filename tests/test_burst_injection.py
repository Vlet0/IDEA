import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluate import _run_pass
from attack import build_trigger, get_semantic_fall_targets

class TestMultiBurstInjection(unittest.TestCase):
    def test_burst_partitioning(self):
        seq_len = 297
        poison_rate = 0.10
        num_bursts = 2
        
        # Total budget should be 30
        total_budget = int(round(seq_len * poison_rate))
        self.assertEqual(total_budget, 30)

        # Check burst segment calculations
        seg_len = seq_len // num_bursts
        burst_lens = []
        rem = total_budget % num_bursts
        ranges = []
        for b in range(num_bursts):
            b_len = total_budget // num_bursts + (1 if b < rem else 0)
            burst_lens.append(b_len)
            seg_start = b * seg_len
            b_start = seg_start + max(0, (seg_len - b_len) // 2)
            b_end = b_start + b_len
            ranges.append((b_start, b_end))

        # Burst 0: [66, 81), Burst 1: [214, 229)
        self.assertEqual(burst_lens, [15, 15])
        self.assertEqual(ranges[0], (66, 81))
        self.assertEqual(ranges[1], (214, 229))
        self.assertGreaterEqual(burst_lens[0], 10, "Burst 0 must clear debounce (>=10)")
        self.assertGreaterEqual(burst_lens[1], 10, "Burst 1 must clear debounce (>=10)")
        self.assertLess(ranges[0][1], ranges[1][0], "Bursts must be non-overlapping with silent gap")

    def test_run_pass_injected_counts(self):
        class DummyNet(torch.nn.Module):
            def forward(self, x):
                B = len(x)
                # Output non-zero only when x is non-zero
                is_nonzero = (x.abs().sum(dim=(1, 2, 3)) > 1e-4).float().unsqueeze(1).unsqueeze(2)
                poses = torch.ones(B, 17, 3) * is_nonzero
                latent = torch.zeros(B, 128)
                return poses, latent

        net = DummyNet()
        K = 1
        trigger = build_trigger("hadamard", K=K, antennas=3, subcarriers=114, packets=10)
        fall_targets = get_semantic_fall_targets(K=K)
        
        x = torch.zeros(297, 3, 114, 10)
        y = torch.zeros(297, 17, 3)
        labels = torch.zeros(297, dtype=torch.bool)
        dataset = TensorDataset(x, y, labels)
        loader = DataLoader(dataset, batch_size=32)

        def trig_idx(b, dev, off):
            return torch.zeros(b, dtype=torch.long, device=dev)

        target_cfg = {"preserve_subject_scale": False, "variation_std": 0.0}
        # Test with num_bursts = 2, poison_rate = 0.10
        r2 = _run_pass(
            net, loader, trigger, eps=0.35, device="cpu",
            fall_targets=fall_targets, target_cfg=target_cfg, trigger_index=trig_idx,
            U=None, poison_rate=0.10, poison_seed=42, frames_per_seq=297,
            num_bursts=2
        )
        injected_frames = (r2["trig_pred"].abs().sum(dim=(1, 2)) > 1e-4).sum().item()
        self.assertEqual(injected_frames, 30, "Exactly 30 frames must be injected for 10% rate on 297 frames")

        # Test with num_bursts = 1 (single continuous burst of 30)
        r1 = _run_pass(
            net, loader, trigger, eps=0.35, device="cpu",
            fall_targets=fall_targets, target_cfg=target_cfg, trigger_index=trig_idx,
            U=None, poison_rate=0.10, poison_seed=42, frames_per_seq=297,
            num_bursts=1
        )
        injected_frames_1 = (r1["trig_pred"].abs().sum(dim=(1, 2)) > 1e-4).sum().item()
        self.assertEqual(injected_frames_1, 30)

        # Test with poison_rate = 1.0 (all frames injected)
        r_all = _run_pass(
            net, loader, trigger, eps=0.35, device="cpu",
            fall_targets=fall_targets, target_cfg=target_cfg, trigger_index=trig_idx,
            U=None, poison_rate=1.0, poison_seed=42, frames_per_seq=297,
            num_bursts=2
        )
        injected_frames_all = (r_all["trig_pred"].abs().sum(dim=(1, 2)) > 1e-4).sum().item()
        self.assertEqual(injected_frames_all, 297)

if __name__ == "__main__":
    unittest.main()

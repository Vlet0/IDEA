"""Fixed, seeded training-set poisoning."""

import unittest

import torch
from torch.utils.data import TensorDataset

from attack import PoisonedDataset, build_poison_plan, build_trigger, dataset_tensors


class PoisoningTests(unittest.TestCase):
    def test_plan_is_deterministic_exact_and_balanced(self):
        a = build_poison_plan(1000, 0.05, 2, seed=7)
        b = build_poison_plan(1000, 0.05, 2, seed=7)
        c = build_poison_plan(1000, 0.05, 2, seed=8)
        self.assertTrue(torch.equal(a.mask, b.mask) and torch.equal(a.kk, b.kk))
        self.assertFalse(torch.equal(a.mask, c.mask))
        self.assertEqual(a.n_poisoned, 50)
        counts = a.per_trigger_counts(2)
        self.assertEqual(sum(counts), 50)
        self.assertLessEqual(abs(counts[0] - counts[1]), 1)

    def test_dataset_poisons_exactly_the_planned_samples_once_and_statically(self):
        torch.manual_seed(0)
        X, Y = torch.rand(10, 3, 114, 10), torch.zeros(10, 17, 3)
        plan = build_poison_plan(10, 0.3, 2, seed=0)
        trigger = build_trigger("hadamard", K=2, antennas=3, subcarriers=114, packets=10)
        data = PoisonedDataset(X, Y, plan, trigger, eps=0.15)
        self.assertEqual(len(data), 10)
        for i in range(10):
            x, y, pm, k = data[i]
            self.assertEqual(bool(pm), bool(plan.mask[i]))
            if plan.mask[i]:
                expected = trigger.apply(X[i:i + 1], int(plan.kk[i]), 0.15)[0]
                self.assertTrue(torch.allclose(x, expected))
                self.assertTrue(torch.equal(x, data[i][0]))  # same frame on every epoch
            else:
                self.assertTrue(torch.equal(x, X[i]))
            self.assertTrue(torch.equal(y, Y[i]))

    def test_dataset_tensors_accepts_tensor_datasets(self):
        X, Y = torch.zeros(3, 3, 114, 10), torch.zeros(3, 17, 3)
        x, y = dataset_tensors(TensorDataset(X, Y))
        self.assertIs(x, X)
        self.assertIs(y, Y)


if __name__ == "__main__":
    unittest.main()

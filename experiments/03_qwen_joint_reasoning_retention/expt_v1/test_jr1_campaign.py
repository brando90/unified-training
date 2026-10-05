"""CPU-only invariants for the JR1 objective primitives."""
import os
import tempfile

os.environ.setdefault("JR1_STATE_ROOT", tempfile.mkdtemp(prefix="jr1-test-"))

import unittest

import torch

import jr1_campaign as j


class JR1ObjectiveTest(unittest.TestCase):
    def test_answer_parser_and_exact_reward(self):
        self.assertEqual(j.extract_answer("work #### 8/2"), "4")
        self.assertIsNone(j.extract_answer("no number"))
        self.assertEqual(j.exact_reward("work #### 4", "answer #### 4"), 1.0)
        self.assertEqual(j.exact_reward("work #### 5", "answer #### 4"), 0.0)


    def test_prompt_mask_preserves_only_completion(self):
        ids = torch.tensor([[9, 1, 2, 0], [9, 3, 4, 5]])
        labels = j.mask_labels(ids, [2, 1], 0)
        self.assertEqual(labels.tolist(), [[-100, -100, 2, -100], [-100, 3, 4, 5]])


    def test_sequence_summed_dpo_prefers_positive_margin(self):
        better = j.dpo_loss(torch.tensor([3.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        worse = j.dpo_loss(torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        self.assertLess(better, worse)


    def test_leave_one_out_advantages_and_invalid_group(self):
        result = j.loo_advantages(torch.tensor([0.0, 1.0, 0.0, 1.0]), 2)
        self.assertTrue(torch.equal(result, torch.tensor([-1.0, 1.0, -1.0, 1.0])))
        with self.assertRaises(ValueError):
            j.loo_advantages(torch.tensor([0.0, 1.0, 1.0]), 2)


if __name__ == "__main__":
    unittest.main()

"""Run with: python3 -m unittest discover -s experiments/00_program -v."""

import json
import math
import unittest

from controller import FIXED, METHODS, OBJECTIVES, SMOOTH_END, SMOOTH_START
from controller import ObjectiveScheduler, validation_progress


def values(*items):
    return dict(zip(OBJECTIVES, items))


class SchedulerTests(unittest.TestCase):
    def assert_simplex(self, probabilities, floor=0.0):
        self.assertEqual(set(probabilities), set(OBJECTIVES))
        self.assertAlmostEqual(math.fsum(probabilities.values()), 1.0, places=14)
        self.assertTrue(all(floor <= x <= 1 for x in probabilities.values()))

    def test_baselines_and_all_objective_support(self):
        for method in METHODS:
            controller = ObjectiveScheduler(method)
            for progress in (0, 0.01, 0.55, 0.75, 0.85, 1):
                probabilities = controller.probabilities(progress)
                self.assert_simplex(probabilities)
                if method != "sequential":
                    self.assertTrue(all(x > 0 for x in probabilities.values()))
        self.assertEqual(ObjectiveScheduler("fixed").probabilities(), dict(zip(OBJECTIVES, FIXED)))
        self.assertEqual(ObjectiveScheduler("uniform").probabilities(), values(.25, .25, .25, .25))

    def test_sequential_boundaries(self):
        controller = ObjectiveScheduler("sequential")
        for progress, expected in ((0, "pt"), (.549999, "pt"), (.55, "sft"),
                                   (.749999, "sft"), (.75, "dpo"), (.849999, "dpo"),
                                   (.85, "rl"), (1, "rl")):
            self.assertEqual(controller.probabilities(progress)[expected], 1)
            self.assertEqual(controller.sample(progress), expected)

    def test_smooth_endpoints_and_midpoint(self):
        controller = ObjectiveScheduler("smooth")
        self.assertEqual(controller.probabilities(0), dict(zip(OBJECTIVES, SMOOTH_START)))
        self.assertEqual(controller.probabilities(1), dict(zip(OBJECTIVES, SMOOTH_END)))
        for i, probability in enumerate(controller.probabilities(.5).values()):
            self.assertAlmostEqual(probability, (SMOOTH_START[i] + SMOOTH_END[i]) / 2)

    def test_floor_extreme_feedback_and_no_nan(self):
        controller = ObjectiveScheduler("validation_progress", eta=1000)
        for dominant in OBJECTIVES:
            rewards = {key: 1.0 if key == dominant else -1.0 for key in OBJECTIVES}
            for _ in range(10):
                self.assert_simplex(controller.update(rewards, values(1, 1, 1, 1)), floor=.02)
        self.assertEqual(controller.snapshot()["updates"], 40)

    def test_cost_adjustment_and_accounting(self):
        cheap = ObjectiveScheduler("validation_progress")
        expensive = ObjectiveScheduler("validation_progress")
        cheap.update(values(0, 0, 0, 1), values(1, 1, 1, 1))
        expensive.update(values(0, 0, 0, 1), values(1, 1, 1, 10))
        self.assertGreater(cheap.probabilities()["rl"], expensive.probabilities()["rl"])
        expensive.record_cost("pt", 7)
        self.assertEqual(expensive.total_cost, 20)
        self.assertEqual(expensive.snapshot()["last_update"]["utilities"]["rl"], .1)

    def test_equal_cost_normalized_rewards_leave_weights_unchanged(self):
        controller = ObjectiveScheduler("validation_progress")
        initial = controller.probabilities()
        result = controller.update(values(1, 2, 3, 4), values(1, 2, 3, 4))
        for key in OBJECTIVES:
            self.assertAlmostEqual(initial[key], result[key])

    def test_cost_units_and_frozen_utility_clipping(self):
        normalized = ObjectiveScheduler("validation_progress")
        raw_counts = ObjectiveScheduler("validation_progress")
        normalized.update(values(0, 0, 0, 1), values(1, 1, 1, 1))
        raw_counts.update(values(0, 0, 0, 1), values(1e6, 1e6, 1e6, 1e6))
        self.assertGreater(normalized.probabilities()["rl"], raw_counts.probabilities()["rl"])
        self.assertAlmostEqual(raw_counts.snapshot()["last_update"]["utilities"]["rl"], 1e-6)
        clipped = ObjectiveScheduler("validation_progress", utility_clip=2)
        clipped.update(values(-100, 0, 0, 100), values(1, 1, 1, 1))
        self.assertEqual(clipped.snapshot()["last_update"]["raw_utilities"]["rl"], 100)
        self.assertEqual(clipped.snapshot()["last_update"]["utilities"]["rl"], 2)
        self.assertEqual(clipped.snapshot()["last_update"]["utilities"]["pt"], -2)

    def test_negative_reward_decreases_weight(self):
        controller = ObjectiveScheduler("validation_progress")
        initial = controller.probabilities()["rl"]
        controller.update(values(0, 0, 0, -1), values(1, 1, 1, 1))
        self.assertLess(controller.probabilities()["rl"], initial)

    def test_rng_reproducibility_and_seed_effect(self):
        first = ObjectiveScheduler("uniform", seed=27)
        second = ObjectiveScheduler("uniform", seed=27)
        third = ObjectiveScheduler("uniform", seed=28)
        sequence = [first.sample() for _ in range(100)]
        self.assertEqual(sequence, [second.sample() for _ in range(100)])
        self.assertNotEqual(sequence, [third.sample() for _ in range(100)])
        self.assertEqual(first.snapshot()["draws"], 100)

    def test_retention_initial_constraint_and_best_diagnostic(self):
        controller = ObjectiveScheduler("validation_progress", pt_initial_loss=10)
        controller.update(values(-5, 5, 5, 5), values(1, 1, 1, 1), pt_validation_loss=8)
        controller.update(values(-5, 5, 5, 5), values(1, 1, 1, 1), pt_validation_loss=9)
        self.assertFalse(controller.snapshot()["last_update"]["retention_active"])
        self.assertAlmostEqual(controller.snapshot()["retention"]["relative_best_degradation"], .125)
        self.assertLess(controller.probabilities()["pt"], .55)
        result = controller.update(values(-5, 5, 5, 5), values(1, 1, 1, 1), pt_validation_loss=10.3)
        self.assertGreaterEqual(result["pt"], .55)
        self.assert_simplex(result, .02)
        self.assertTrue(controller.snapshot()["last_update"]["retention_active"])

    def test_invalid_inputs_do_not_mutate_state(self):
        controller = ObjectiveScheduler("validation_progress")
        original = controller.snapshot()
        cases = [
            ({"pt": 1}, values(1, 1, 1, 1), {}),
            (values(0, 0, 0, float("nan")), values(1, 1, 1, 1), {}),
            (values(0, 0, 0, 1), values(1, 1, 1, 0), {}),
            (values(0, 0, 0, 1), values(1, 1, 1, -1), {}),
            (values(0, 0, 0, 1), values(1, 1, 1, float("inf")), {}),
            (values(0, 0, 0, True), values(1, 1, 1, 1), {}),
            (values(0, 0, 0, 1), values(1, 1, 1, 1), {"pt_validation_loss": 1}),
            (values(0, 0, 0, 1e308), values(1, 1, 1, 1e-308), {}),
            (values(0, 0, 0, 0), values(1e308, 1e308, 1e308, 1e308), {}),
        ]
        for rewards, costs, kwargs in cases:
            with self.assertRaises(ValueError):
                controller.update(rewards, costs, **kwargs)
            self.assertEqual(controller.snapshot(), original)
        for progress in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                controller.sample(progress)
            self.assertEqual(controller.snapshot(), original)
        for bad_cost in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                controller.record_cost("pt", bad_cost)
            self.assertEqual(controller.snapshot(), original)

    def test_invalid_configuration_and_baseline_updates(self):
        for kwargs in ({"floor": 0}, {"floor": .25}, {"eta": -1}, {"seed": 1.2},
                       {"retention_min_pt": 1}, {"pt_initial_loss": 0},
                       {"retention_threshold": -1}):
            with self.assertRaises(ValueError):
                ObjectiveScheduler("validation_progress", **kwargs)
        with self.assertRaises(ValueError):
            ObjectiveScheduler("aioli")
        with self.assertRaises(ValueError):
            ObjectiveScheduler("fixed").update(values(1, 1, 1, 1), values(1, 1, 1, 1))

    def test_retention_exact_threshold_is_not_breached(self):
        controller = ObjectiveScheduler("validation_progress", pt_initial_loss=10)
        controller.update(values(-5, 5, 5, 5), values(1, 1, 1, 1), pt_validation_loss=10.2)
        self.assertFalse(controller.snapshot()["last_update"]["retention_active"])

    def test_retention_overflow_does_not_mutate_state(self):
        controller = ObjectiveScheduler("validation_progress", pt_initial_loss=1e-308)
        before = controller.snapshot()
        with self.assertRaises(ValueError):
            controller.update(values(0, 0, 0, 0), values(1, 1, 1, 1), pt_validation_loss=1e308)
        self.assertEqual(controller.snapshot(), before)

    def test_snapshot_copies_and_json_serialization(self):
        controller = ObjectiveScheduler("validation_progress")
        controller.update(values(1, 1, 1, 1), values(1, 1, 1, 1))
        snapshot = controller.snapshot()
        json.dumps(snapshot, allow_nan=False)
        snapshot["last_update"]["costs"]["pt"] = -100
        snapshot["adaptive_weights"]["pt"] = -100
        self.assertEqual(controller.snapshot()["last_update"]["costs"]["pt"], 1)
        self.assertGreater(controller.probabilities()["pt"], 0)


class ValidationProgressTests(unittest.TestCase):
    def test_relative_scaling_and_degradation(self):
        initial = {"language": 1000, "instruction": 1}
        before = {"language": 800, "instruction": .8}
        after = {"language": 700, "instruction": .9}
        self.assertAlmostEqual(validation_progress(initial, before, after), 0)
        self.assertAlmostEqual(validation_progress(initial, before, {"language": 900, "instruction": .9}), -.1)

    def test_validation_input_validation(self):
        cases = [({}, {}, {}), ({"a": 0}, {"a": 1}, {"a": 1}),
                 ({"a": 1}, {"a": 1}, {"b": 1}),
                 ({"a": 1}, {"a": float("nan")}, {"a": 1}),
                 ({"a": 1}, {"a": 1}, {"a": -1})]
        for args in cases:
            with self.assertRaises(ValueError):
                validation_progress(*args)


if __name__ == "__main__":
    unittest.main()

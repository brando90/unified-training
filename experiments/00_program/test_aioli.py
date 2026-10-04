"""Deterministic equation checks, independent of a training run."""

import json
import math
import unittest

from aioli import AioliController, normalize_transfer_matrix, recover_transfer_matrix, smoothed_probe_matrix


class AioliTests(unittest.TestCase):
    def test_unequal_cost_column_division_reverses_true_effects(self):
        from aioli import smoothed_probe_matrix, recover_transfer_matrix
        W=smoothed_probe_matrix();true=[1.,2.,0.,0.];costs=[1.,5.,1.,1.]
        drops=[sum(w*a for w,a in zip(row,true)) for row in W]
        probe_costs=[12*sum(w*c for w,c in zip(row,costs)) for row in W]
        self.assertEqual(probe_costs,[16.,48.,16.,16.])
        recovered=recover_transfer_matrix([drops])[0]
        self.assertLess(max(abs(a-b) for a,b in zip(recovered,true)),1e-12)
        broken=recover_transfer_matrix([[d/c for d,c in zip(drops,probe_costs)]])[0]
        self.assertGreater(recovered[1],recovered[0]);self.assertGreater(broken[0],broken[1])

    def setUp(self):
        # Three validation components, four actions; deliberately nondiagonal.
        self.A = [[.2, -.1, .7, .4], [-.4, .9, .3, .2], [.8, .6, -.2, .1]]

    def drops(self, matrix, alpha=.75):
        W = smoothed_probe_matrix(4, alpha)
        return [[math.fsum(W[j][k] * row[k] for k in range(4)) for j in range(4)] for row in matrix]

    def test_recovers_cross_influences_with_arbitrary_components(self):
        for alpha in (.75, 1, .1):
            recovered = recover_transfer_matrix(self.drops(self.A, alpha), alpha=alpha)
            for expected_row, actual_row in zip(self.A, recovered):
                for expected, actual in zip(expected_row, actual_row):
                    self.assertAlmostEqual(expected, actual, places=12)

    def test_global_shift_and_normalization(self):
        normalized = normalize_transfer_matrix(self.A)
        total = sum(x + .4 for row in self.A for x in row)
        self.assertAlmostEqual(sum(map(sum, normalized)), 1)
        for i, row in enumerate(self.A):
            for j, value in enumerate(row):
                self.assertAlmostEqual(normalized[i][j], (value + .4) / total)

    def test_exact_multiplicative_update(self):
        controller = AioliController(eta=2, initial_weights=[.4, .3, .2, .1])
        normalized = normalize_transfer_matrix(self.A)
        unnormalized = [w * math.exp(2 * sum(row[j] for row in normalized))
                        for j, w in enumerate([.4, .3, .2, .1])]
        expected = [x / sum(unnormalized) for x in unnormalized]
        actual = list(controller.update(self.drops(self.A)).values())
        for x, y in zip(expected, actual):
            self.assertAlmostEqual(x, y, places=14)
        self.assertTrue(all(x > 0 for x in actual))
        self.assertAlmostEqual(sum(actual), 1)

    def test_zero_matrix_and_constant_negative_are_neutral(self):
        for value in (0, -2):
            controller = AioliController(initial_weights=[.4, .3, .2, .1])
            before = controller.probabilities()
            after = controller.update([[value] * 4 for _ in range(3)])
            for key in before:
                self.assertAlmostEqual(before[key], after[key])

    def test_rounds_average_before_update(self):
        first, second = AioliController(), AioliController()
        drops = self.drops(self.A)
        low = [[x - .03 for x in row] for row in drops]
        high = [[x + .03 for x in row] for row in drops]
        expected = first.update(drops)
        actual = second.update_rounds([low, high])
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key])

    def test_probe_randomization_and_sampling_are_repeatable(self):
        first, second = AioliController(seed=12), AioliController(seed=12)
        orders = first.probe_order(3)
        self.assertEqual(orders, second.probe_order(3))
        for i in range(3):
            self.assertEqual(sorted(orders[i * 4: (i + 1) * 4]), list(range(4)))
        self.assertEqual([first.sample() for _ in range(30)], [second.sample() for _ in range(30)])
        for i, mixture in enumerate(first.probe_mixtures()):
            self.assertAlmostEqual(sum(mixture.values()), 1)
            self.assertEqual(list(mixture.values())[i], .75)

    def test_optional_floor_is_explicit_adaptation(self):
        controller = AioliController(eta=10000, floor=.02)
        weights = controller.update(self.drops(self.A))
        self.assertTrue(all(x >= .02 for x in weights.values()))
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertTrue(controller.snapshot()["floor_is_adaptation"])

    def test_invalid_shapes_nonfinite_singular_and_atomicity(self):
        controller = AioliController()
        initial = controller.snapshot()
        for matrix in ([], [[1, 2, 3]], [[1, 2, 3, float("nan")]], [[1, 2, 3, float("inf")]]):
            with self.assertRaises(ValueError):
                controller.update(matrix)
            self.assertEqual(initial, controller.snapshot())
        for kwargs in ({"alpha": .25}, {"alpha": 2}, {"alpha": float("nan")},
                       {"floor": .25}, {"eta": 0}, {"initial_weights": [1, 0, 1, 1]}):
            with self.assertRaises(ValueError):
                AioliController(**kwargs)
        with self.assertRaises(ValueError):
            controller.update_rounds([])
        controller.update([[0] * 4])
        with self.assertRaises(ValueError):
            controller.update([[0] * 4, [0] * 4])

    def test_receipt_is_finite_json_and_copied(self):
        controller = AioliController()
        controller.update(self.drops(self.A))
        snapshot = controller.snapshot()
        json.dumps(snapshot, allow_nan=False)
        snapshot["last_update"]["transfer"][0][0] = 999
        self.assertAlmostEqual(controller.snapshot()["last_update"]["transfer"][0][0], .2)


if __name__ == "__main__":
    unittest.main()

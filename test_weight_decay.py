from dataclasses import replace
import math
import unittest

import numpy as np

from baseline import Config, fit, make_problem
from weight_decay import DECAYS, METRICS, PREFIX, advance_above_threshold, crossings, summarize, theory, trial_configs


class WeightDecayTests(unittest.TestCase):
    def test_official_order_and_seeds(self):
        configs = list(trial_configs(Config.load()))
        expected = [2e-6, 4e-6, 8e-6, 16e-6, 32e-6]
        self.assertEqual(len(configs), 50)
        for index, (trial, c) in enumerate(configs):
            self.assertEqual((trial, c.weight_decay), (index % 10 + 1, expected[index // 10]))
            self.assertEqual((c.teacher_seed, c.train_seed, c.test_seed, c.student_seed),
                             (1000*index, 1000*index + 1, 1000*index + 2, 100))

    def test_theory_on_known_spectrum(self):
        c = replace(Config.load(), n_train=2, dimension=3, weight_decay=0.001)
        spectrum, upper, lower = theory(c, np.array([[2., 0., 0.], [0., 3., 0.]]))
        self.assertEqual(spectrum, 4.)
        self.assertAlmostEqual(upper, 2 * math.log(4200.) / 8)
        self.assertAlmostEqual(lower, math.log(1 / 0.24) / 0.00202)

    def test_fast_forward_and_crossings_against_literal_gd(self):
        c = replace(Config.load(), n_train=4, dimension=16, n_test=24, weight_decay=0.0003,
                    epsilon=0.15, max_steps=20000)
        _, train, test, initial = make_problem(c)
        prefix_state, _, _, _ = fit(replace(c, max_steps=PREFIX), train, test, initial)
        skipped, jumped = advance_above_threshold(c, train, test, prefix_state, c.max_steps-PREFIX)
        self.assertGreater(skipped, 100)
        literal, trace, _, _ = fit(replace(c, max_steps=skipped), train, test, prefix_state)
        np.testing.assert_allclose(jumped, literal, rtol=1e-10, atol=1e-12)
        self.assertTrue(np.all(trace[:, 1] > c.epsilon))
        t1, t2, _ = crossings(c, train, test, initial, literal=True)
        self.assertTrue(t1 < PREFIX < t2)
        for cap in (t2-1, t2, c.max_steps):
            with self.subTest(cap=cap):
                result = crossings(replace(c, max_steps=cap), train, test, initial)
                self.assertEqual(result[:2], (t1, t2 if cap >= t2 else None))

    def test_summary_uses_paired_delays_and_population_sd(self):
        rows = []
        for decay in DECAYS:
            for i in range(10):
                rows.append(dict(weight_decay=decay, t1=i, t2=i+20, delay=20, t1_upper=10, t2_lower=15))
        for group in summarize(rows):
            self.assertEqual(group["t2_mean"], 24.5)
            self.assertAlmostEqual(group["t1_sd"], math.sqrt(8.25))
            self.assertEqual((group["delay_mean"], group["delay_sd"]), (20, 0))
            for metric in METRICS:
                self.assertIn(metric + "_mean", group)


if __name__ == "__main__":
    unittest.main()

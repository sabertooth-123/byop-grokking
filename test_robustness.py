from dataclasses import replace
import unittest

import numpy as np

from baseline import Config, fit, make_problem, population_mse
from robustness import EPSILONS, MISSING, TEST_OFFSETS, aggregate, differences, finite_set, measure, trial_config


class RobustnessTests(unittest.TestCase):
    def test_seeds_pair_within_trial_and_are_distinct_across_trials(self):
        seeds = []
        for j in range(20):
            c = trial_config(Config.load(), j+1)
            self.assertEqual((c.teacher_seed, c.train_seed, c.student_seed, c.test_seed),
                             (1000*j, 1000*j+1, 1000*j+100, 1000*j+2))
            seeds.extend([c.teacher_seed, c.train_seed, c.student_seed])
            seeds.extend(1000*j+offset for offset in TEST_OFFSETS.values())
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_exact_population_design_and_finite_split(self):
        c = replace(Config.load(), dimension=7, n_train=3, n_test=19)
        teacher, _, test, initial = make_problem(c)
        x, y = finite_set(c, teacher, c.n_test, c.test_seed)
        np.testing.assert_array_equal(x, test[0])
        np.testing.assert_array_equal(y, test[1])
        for w in (initial, teacher, initial * 2):
            self.assertAlmostEqual(np.mean((np.eye(7) @ w-teacher)**2), population_mse(w, teacher))

    def test_observed_and_capped_runs_against_direct_gd(self):
        c = replace(Config.load(), dimension=64, n_train=8, n_test=200, max_steps=50000)
        teacher, train, finite, initial = make_problem(c)
        for evaluation in [(np.eye(c.dimension), teacher), finite]:
            for e, cap in [(EPSILONS[-1], c.max_steps), (EPSILONS[0], 100)]:
                config = replace(c, epsilon=e, max_steps=cap)
                _, _, t1, t2 = fit(config, train, evaluation, initial)
                result = measure(config, train, evaluation, initial)
                self.assertEqual(result["t1"], MISSING if t1 is None else t1)
                self.assertEqual(result["t2"], MISSING if t2 is None else t2)
                if cap == 100:
                    self.assertEqual(result["delay"], MISSING)
                else:
                    self.assertIsNotNone(t2)
                    self.assertEqual(result["delay"], t2-t1)
                    self.assertGreater(result["skipped_updates"], 0)

    def test_paired_difference_and_missing(self):
        self.assertEqual(differences(12, 16), (-4, 4))
        self.assertEqual(differences(20, 16), (4, 4))
        self.assertEqual(differences(16, 16), (0, 0))
        self.assertEqual(differences(12, MISSING), (MISSING, MISSING))

    def test_summary_and_trial_bootstrap(self):
        rows = [dict(trial=i, epsilon=e, t1=i+e, t2=100+i+e, delay=100)
                for e in (1, 2) for i in range(1, 6)]
        stats = aggregate(rows, "epsilon", (1, 2))
        t1, _, delay = stats[:3]
        self.assertEqual((t1["mean"], t1["median"], t1["observed"]), (4, 4, 5))
        self.assertAlmostEqual(t1["sd"], np.sqrt(2.5))
        self.assertEqual((delay["mean_ci_low"], delay["median_ci_high"]), (100, 100))
        for key in ("mean_ci_low", "mean_ci_high", "median_ci_low", "median_ci_high"):
            self.assertAlmostEqual(stats[3][key]-t1[key], 1)
        rows[0].update(t2=MISSING, delay=MISSING)
        censored = aggregate(rows, "epsilon", (1, 2))[1]
        self.assertEqual((censored["observed"], censored["unobserved"]), (4, 1))
        self.assertTrue(np.isnan(censored["mean_ci_low"]))
        self.assertTrue(np.isnan(censored["mean"]))

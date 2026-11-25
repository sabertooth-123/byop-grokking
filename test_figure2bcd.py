from dataclasses import replace
import math
import unittest

import numpy as np

from baseline import Config, make_problem
from figure2bcd import UNOBSERVED, bounds, configs, reference_module, smallest_positive, summarize
from weight_decay import crossings


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = reference_module()

    def test_grids_and_independent_seed_progression(self):
        base = Config.load()
        for panel, field, grid in [
            ("sample_size", "n_train", [10, 20, 40, 80, 160]),
            ("feature_dimension", "dimension", [1000, 1500, 2000, 2500, 3000]),
            ("initialization", "student_variance", [1, 10, 100, 1000, 10000, 100000, 1000000])
        ]:
            rows = list(configs(panel, base))
            self.assertEqual(len(rows), len(grid)*10)
            for i, (value, trial, c) in enumerate(rows):
                with self.subTest(panel=panel, i=i):
                    self.assertEqual((value, trial), (grid[i//10], i%10+1))
                    self.assertEqual(c, replace(base, **{field: grid[i//10]}, teacher_seed=1000*i,
                                                train_seed=1000*i+1, test_seed=1000*i+2))

    def test_scaled_initialization_and_reference_arguments(self):
        c = replace(Config.load(), dimension=20, n_train=3, student_variance=100.)
        _, (x, _), _, w = make_problem(c)
        np.testing.assert_array_equal(w, np.random.default_rng(100).normal(0, 10, size=20))
        eig = self.ref.lambda_min_positive_gram(x)
        for panel in ("sample_size", "feature_dimension", "initialization"):
            result = bounds(self.ref, panel, c, smallest_positive(x))
            arg = 10 if panel == "initialization" else 100
            expected = self.ref.theoretical_upper_bound_t1(3, 20, 1, .01, arg, eig)
            self.assertAlmostEqual(result["t1_upper"], expected)
            self.assertAlmostEqual(result["t2_lower"], math.log(17*100/(8*20*.01))/(2.02*.0001))
            if panel == "initialization":
                self.assertGreater(result["t1_upper_variance"], expected)
                self.assertAlmostEqual(result["t1_upper_variance"], 3*math.log(14*20*100/.01)/(2*eig))

    def test_positive_spectrum_on_rank_deficient_matrix(self):
        x = np.array([[3., 0., 0., 0.], [6., 0., 0., 0.]])
        self.assertAlmostEqual(smallest_positive(x), self.ref.lambda_min_positive_gram(x))

    def test_accelerated_crossings_at_different_shapes_and_scales(self):
        for n, m, variance in [(10, 64, 1), (20, 96, 1), (10, 64, 100)]:
            with self.subTest(n=n, m=m, variance=variance):
                c = replace(Config.load(), n_train=n, dimension=m, n_test=40, student_variance=variance,
                            epsilon=.08, max_steps=60000)
                _, train, test, w = make_problem(c)
                t1, t2, skipped = crossings(c, train, test, w)
                self.assertGreater(skipped, 0)
                self.assertIsNotNone(t2)
                self.assertEqual((t1, t2), crossings(c, train, test, w, literal=True)[:2])

    def test_missing_threshold_propagates(self):
        rows = [dict(value=v, trial=i, t1=i, t2=i+100, delay=100, t1_upper=50, t2_lower=60)
                for v in [10,20,40,80,160] for i in range(1,11)]
        rows[0].update(t2=UNOBSERVED, delay=UNOBSERVED)
        first = summarize("sample_size", rows)[0]
        self.assertEqual(first["t1_mean"], 5.5)
        self.assertAlmostEqual(first["t1_sd"], math.sqrt(8.25))
        for key in ("t2", "delay"):
            self.assertEqual(first[key+"_observed"], 9)
            self.assertTrue(math.isnan(first[key+"_mean"]))
            self.assertTrue(math.isnan(first[key+"_sd"]))

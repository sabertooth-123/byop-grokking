from dataclasses import replace
import unittest

import numpy as np

from baseline import Config, fit, gradient, make_problem


class BaselineTests(unittest.TestCase):
    def test_generator(self):
        c = replace(Config.load(), dimension=6, n_train=3, n_test=8)
        teacher, train, test, initial = make_problem(c)
        self.assertAlmostEqual(np.linalg.norm(teacher), 1.)
        for (x, y), n, seed in [(train, 3, c.train_seed), (test, 8, c.test_seed)]:
            draws = np.random.default_rng(seed).standard_normal((n, 6))
            np.testing.assert_allclose(x, draws / np.sqrt(6), atol=1e-15)
            np.testing.assert_allclose(y, (x * teacher).sum(axis=1), atol=1e-15)
        np.testing.assert_array_equal(initial, np.random.default_rng(100).standard_normal(6))

    def test_gradient_by_finite_differences(self):
        x = np.array([[0.5, 1., 2.], [0., -0.25, 1.]])
        y = np.array([0.3, -0.4])
        w = np.array([0.1, -0.8, 0.7])
        decay = 0.13

        def objective(v):
            return (np.mean((x @ v - y)**2) + decay * (v @ v)) / 2

        h = 1e-6
        numeric = [(objective(w+h*e) - objective(w-h*e))/(2*h) for e in np.eye(3)]
        np.testing.assert_allclose(gradient(x, x @ w-y, w, decay), numeric, atol=1e-10, rtol=1e-9)

    def test_diagonal_recurrence_and_history(self):
        # Observed coordinate shrinks with curvature 1+lambda; null coordinate with lambda.
        c = replace(Config.load(), step_size=0.5, weight_decay=0.25, max_steps=5)
        train = np.array([[1., 0.]]), np.array([0.])
        test = np.array([[0., 1.]]), np.array([0.])
        final, losses, t1, t2 = fit(c, train, test, np.ones(2))
        trajectory = np.array([0.375, 0.875]) ** np.arange(6)[:, None]
        np.testing.assert_allclose(losses, trajectory**2)
        np.testing.assert_allclose(final, trajectory[-1])
        self.assertEqual((t1, t2), (3, None))

    def test_equality_at_initialization(self):
        data = np.ones((1, 1)), np.zeros(1)
        _, losses, t1, t2 = fit(replace(Config.load(), epsilon=1.), data, data, np.ones(1))
        self.assertEqual((len(losses), t1, t2), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()

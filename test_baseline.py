from dataclasses import replace
import unittest

import numpy as np

from baseline import Config, fit, gradient, make_problem, population_mse
from compare_reference import validate_initialization


class BaselineTests(unittest.TestCase):
    def test_deterministic_full_run(self):
        c = Config.load()

        def run():
            _, train, test, initial = make_problem(c)
            return fit(c, train, test, initial)

        final, losses, t1, t2 = run()
        repeat_final, repeat_losses, repeat_t1, repeat_t2 = run()
        np.testing.assert_array_equal(final, repeat_final)
        np.testing.assert_array_equal(losses, repeat_losses)
        self.assertEqual((t1, t2), (repeat_t1, repeat_t2))
        self.assertEqual((t1, t2, len(losses)), (232, 22769, 22770))

    def test_neither_threshold_crosses_before_cap(self):
        c = replace(Config.load(), max_steps=2, step_size=0.1, weight_decay=0.)
        data = np.ones((1, 1)), np.zeros(1)
        final, losses, t1, t2 = fit(c, data, data, np.ones(1))
        self.assertEqual((t1, t2), (None, None))
        np.testing.assert_allclose(final, [0.81])
        expected = np.array([1., 0.9**2, 0.81**2])
        np.testing.assert_allclose(losses, np.column_stack((expected, expected)))

    def test_exact_population_loss(self):
        teacher = np.array([1., 0., 0., 0.])
        weights = np.array([2., -2., 3., 0.])
        # Signed coordinate vectors integrate quadratics exactly for covariance I/4.
        points = np.vstack((np.eye(4), -np.eye(4)))
        expected = np.mean((points @ weights - points @ teacher)**2)
        self.assertEqual(population_mse(weights, teacher), expected)
        self.assertEqual(population_mse(teacher, teacher), 0.)

    def test_reference_initialization(self):
        c = Config.load()
        validate_initialization(c)
        for change in (dict(student_seed=101), dict(student_variance=2.)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "requires student_seed"):
                validate_initialization(replace(c, **change))

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

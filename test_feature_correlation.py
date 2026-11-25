from dataclasses import replace
import unittest

import numpy as np
from baseline import Config, make_problem, gradient
from feature_correlation import RHOS, Trajectory, correlate, population, spectrum, first_crossings


class CorrelationTests(unittest.TestCase):
    def test_exact_covariance_and_marginals(self):
        m = 19
        for rho in RHOS:
            root = correlate(np.eye(m)/np.sqrt(m), rho)
            sigma = ((1-rho)*np.eye(m)+rho*np.ones((m,m)))/m
            np.testing.assert_allclose(root.T @ root, sigma, atol=1e-14)
            eig = np.linalg.eigvalsh(sigma)
            stats = spectrum(m,rho)
            np.testing.assert_allclose(eig, [stats["bulk_eigenvalue"]]*(m-1)+[stats["shared_axis_eigenvalue"]])
            for d in [np.ones(m), np.arange(m)-m//2, np.random.default_rng(42).normal(size=m)]:
                self.assertAlmostEqual(population(d,rho), d @ sigma @ d)

    def test_sample_moments_and_shared_draws(self):
        m = 8
        base = np.random.default_rng(33).normal(0,1/np.sqrt(m),(100000,m))
        np.testing.assert_array_equal(correlate(base,0),base)
        for rho in RHOS:
            x = correlate(base,rho)
            target = ((1-rho)*np.eye(m)+rho)/m
            # Each Gaussian covariance entry has SE <= sqrt(2/N)/m.
            np.testing.assert_allclose(x.T @ x/len(x),target,atol=6*np.sqrt(2/len(x))/m,rtol=0)

    def test_discrete_losses_with_signed_update_factors(self):
        c = replace(Config.load(),dimension=18,n_train=9,n_test=4)
        t,(z,_),_,w = make_problem(c)
        x = correlate(z,.85)
        # Choose a stable step with an oscillating leading mode.
        c = replace(c,step_size=1.7/(np.linalg.norm(x,2)**2/len(x)+c.weight_decay))
        path = Trajectory(x,t,w,c)
        steps = np.arange(120)
        train,pop,_,_ = path.losses(steps,.85)
        for k in steps:
            r = x @ w-x @ t
            self.assertAlmostEqual(train[k], np.mean(r*r))
            self.assertAlmostEqual(pop[k], population(w-t,.85))
            w = w-c.step_size*gradient(x,r,w,c.weight_decay)
        self.assertEqual(first_crossings(path,.85,0.,5),(None,None))
        self.assertEqual(first_crossings(path,.85,1e10,5),(0,0))

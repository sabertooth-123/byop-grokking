"""Regression checks for the two evaluation geometries and archived paired results."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import geometry_diagnostic as gd
from baseline import Config, gradient, make_problem
from robustness import MISSING, trial_config


class GeometryDiagnosticTests(unittest.TestCase):
    def test_teacher_construction(self):
        original = np.array([1., 2., -1., 3.]) / np.sqrt(15)
        axis = np.full(4, .5)
        group = gd.teachers(original)
        np.testing.assert_array_equal(group["random"], original)
        np.testing.assert_allclose(group["parallel"], axis)
        projected = original - (axis @ original)*axis
        np.testing.assert_allclose(group["orthogonal"], projected/np.linalg.norm(projected))
        for teacher in group.values():
            self.assertAlmostEqual(teacher @ teacher, 1.)
        self.assertAlmostEqual(axis @ group["orthogonal"], 0.)

    def test_metrics_and_first_hits_on_one_literal_trajectory(self):
        c = replace(Config.load(), dimension=8, n_train=4, n_test=3, max_steps=800,
                    weight_decay=.02, epsilon=.25)
        target, (z, _), _, initial = make_problem(c)
        steps = np.arange(c.max_steps+1)
        for rho in (0., .6, .85):
            with self.subTest(rho=rho), patch.object(gd, "CHECKPOINTS", steps):
                x = gd.correlate(z, rho)
                sigma = ((1-rho)*np.eye(c.dimension)+rho)/c.dimension
                counts, pop, iso = gd.evaluate(x, target, initial, c, rho)
                w = initial.copy()
                losses = []
                for _ in steps:
                    d = w-target
                    r = x @ d
                    losses.append((np.mean(r*r), d @ sigma @ d, d @ d/c.dimension))
                    w -= c.step_size*gradient(x, r, w, c.weight_decay)
                literal = np.asarray(losses)
                np.testing.assert_allclose(pop, literal[:,1], rtol=1e-11)
                np.testing.assert_allclose(iso, literal[:,2], rtol=1e-11)
                if rho == 0:
                    np.testing.assert_array_equal(pop, iso)
                for i,key in enumerate(("t1","t_pop","t_iso")):
                    hits = np.flatnonzero(literal[:,i] <= c.epsilon)
                    self.assertTrue(hits.size)
                    self.assertEqual(counts[key], hits[0])
                self.assertEqual(counts["delay_iso"], counts["t_iso"]-counts["t1"])
                self.assertEqual(counts["metric_gap"], counts["t_pop"]-counts["t_iso"])
        for eps, expected in [(0., MISSING), (1e4, 0)]:
            counts, _, _ = gd.evaluate(x,target,initial,replace(c,max_steps=0,epsilon=eps),rho)
            self.assertTrue(all(v == expected for v in counts.values()))

    def test_archived_seed_replay(self):
        c = trial_config(Config.load(), 2)
        teacher,(z,_),_,initial = make_problem(c)
        for r in gd.read_rows(gd.OUT/"trials.csv"):
            if r["trial"]=="2" and float(r["rho"])==.85:
                counts, _, _ = gd.evaluate(gd.correlate(z,.85),gd.teachers(teacher)[r["teacher"]],initial,c,.85)
                for key,val in counts.items():
                    self.assertEqual(str(val),r[key])

    def test_paired_contrasts_and_saved_bootstrap(self):
        rows = []
        for text in gd.read_rows(gd.OUT/"trials.csv"):
            r = {k:(v if k=="teacher" or v=="NA" else float(v)) for k,v in text.items()}
            rows.append(r)
        by_key = {(r["teacher"],r["rho"],r["trial"]):r for r in rows}
        self.assertEqual(len(by_key),128)
        for r in rows:
            control = by_key[r["teacher"],0.,r["trial"]]
            for m in ("pop","iso"):
                delay = r["t_"+m]-r["t1"]
                self.assertEqual(r["delay_"+m],delay)
                self.assertEqual(r["delta_delay_"+m], delay-control["delay_"+m])
            if r["teacher"]=="parallel":
                other = by_key["orthogonal",r["rho"],r["trial"]]
                for key in ("t1","t_pop","t_iso"):
                    self.assertEqual(r["parallel_minus_orthogonal_"+key],r[key]-other[key])
        stats = gd.summarize(rows)
        self.assertEqual(stats,gd.summarize(list(reversed(rows))))
        for computed,saved in zip(stats,gd.read_rows(gd.OUT/"summary.csv"),strict=True):
            for key in computed:
                if key in ("teacher","metric"):
                    self.assertEqual(computed[key],saved[key])
                else:
                    self.assertAlmostEqual(computed[key],float(saved[key]),places=8)

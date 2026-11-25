"""Full-batch ridge regression, Section 5.1 of arXiv:2601.19791."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    n_train: int
    n_test: int
    dimension: int
    teacher_seed: int
    train_seed: int
    test_seed: int
    student_seed: int
    student_variance: float
    step_size: float
    weight_decay: float
    epsilon: float
    max_steps: int

    @classmethod
    def load(cls):
        return cls(**json.loads((ROOT / "baseline.json").read_text()))


def make_problem(c):
    teacher = np.random.default_rng(c.teacher_seed).standard_normal(c.dimension)
    teacher /= np.linalg.norm(teacher)

    def draw_split(size, seed):
        features = np.random.default_rng(seed).normal(
            scale=1 / np.sqrt(c.dimension), size=(size, c.dimension)
        )
        return features, features @ teacher

    student = np.random.default_rng(c.student_seed).normal(
        scale=np.sqrt(c.student_variance), size=c.dimension
    )
    return teacher, draw_split(c.n_train, c.train_seed), draw_split(c.n_test, c.test_seed), student


def gradient(features, residual, weights, decay):
    return (features.T @ residual) / len(residual) + decay * weights


def fit(c, train, test, initial):
    weights = initial.copy()
    x, y = train
    test_x, test_y = test
    history = []
    t1 = t2 = None
    for step in range(c.max_steps + 1):
        residual = x @ weights - y
        train_loss = np.mean(np.square(residual))
        test_loss = np.mean(np.square(test_x @ weights - test_y))
        history.append((train_loss, test_loss))
        if t1 is None and train_loss <= c.epsilon:
            t1 = step
        if t2 is None and test_loss <= c.epsilon:
            t2 = step
        if step == c.max_steps or (step > 0 and t1 is not None and t2 is not None):
            break
        weights -= c.step_size * gradient(x, residual, weights, c.weight_decay)
    return weights, np.array(history), t1, t2

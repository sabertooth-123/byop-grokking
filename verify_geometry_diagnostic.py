"""Check diagnostic endpoints and normalized checkpoints against literal GD."""
import json
from time import perf_counter

import numpy as np

from baseline import Config, gradient, make_problem, population_mse
from feature_correlation import correlate, population
from geometry_diagnostic import OUT, read_rows, teachers
from robustness import MISSING, trial_config


def main():
    start = perf_counter()
    records = read_rows(OUT/"trials.csv")
    checkpoints = read_rows(OUT/"checkpoints.csv")
    checks = []
    for trial,rho,kind in [(1,0.,"random"),(20,.85,"random"),(2,.3,"parallel"),
                          (2,.3,"orthogonal"),(6,.85,"parallel"),(6,.85,"orthogonal")]:
        def matches(r):
            return (int(r["trial"]),float(r["rho"]),r["teacher"])==(trial,rho,kind)
        saved, = [r for r in records if matches(r)]
        trace = {int(r["step"]):r for r in checkpoints if matches(r)}
        c = trial_config(Config.load(),trial)
        original,(z,_),_,w = make_problem(c)
        target = teachers(original)[kind]
        np.testing.assert_allclose(target @ target,1.,atol=1e-14)
        if kind!="random":
            np.testing.assert_allclose(target.sum()/np.sqrt(c.dimension), float(kind=="parallel"),atol=1e-14)
        x = correlate(z,rho)
        y = x @ target
        first = dict.fromkeys(("t1","t_pop","t_iso"),MISSING)
        max_error = 0.
        for step in range(c.max_steps+1):
            residual = x @ w-y
            p = population(w-target,rho)
            i = population_mse(w,target)
            if step==0:
                p0,i0 = p,i
            for key,loss in zip(first,(np.mean(residual**2),p,i)):
                if first[key]==MISSING and loss<=c.epsilon:
                    first[key] = step
            if step in trace:
                actual = np.array([p,i,p/p0,i/i0])
                expected = [float(trace[step][k]) for k in ("population","isotropic","normalized_pop","normalized_iso")]
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)
                max_error = max(max_error, np.max(abs(actual-expected)))
            if step>=max(trace) and MISSING not in first.values():
                break
            w -= c.step_size*gradient(x,residual,w,c.weight_decay)
        for k,v in first.items():
            np.testing.assert_equal(str(v),saved[k])
        checks.append(dict(trial=trial,rho=rho,teacher=kind,**first,steps=step,max_abs_loss_error=max_error))
    report = dict(checks=checks,seconds=perf_counter()-start)
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    main()

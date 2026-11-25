"""Verify saved crossings using literal baseline GD and the Milestone 3 isotropic control."""
import csv
import json
from time import perf_counter

import numpy as np
from baseline import Config, ROOT, gradient, make_problem
from feature_correlation import OUT, RHOS, correlate, population
from robustness import trial_config


def main():
    start = perf_counter()
    with (OUT/"trials.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (ROOT/"results/robustness/population_trials.csv").open() as f:
        controls = {r["trial"]:r for r in csv.DictReader(f) if r["evaluation"]=="population"}
    baseline = {r["trial"]:r for r in rows if float(r["rho"])==0}
    np.testing.assert_equal(set(controls),set(baseline))
    for trial,r in controls.items():
        for key in ("t1","t2","delay"):
            np.testing.assert_equal(baseline[trial][key],r[key])
    checks = []
    with (OUT/"losses.csv").open() as f:
        traces = [r for r in csv.DictReader(f) if int(r["trial"])==1]
    c = trial_config(Config.load(),1)
    teacher,(base,_),_,initial = make_problem(c)
    for rho in RHOS:
        x = correlate(base,rho)
        y = x @ teacher
        target = next(r for r in rows if r["trial"]=="1" and float(r["rho"])==rho)
        samples = {int(r["step"]):r for r in traces if float(r["rho"])==rho}
        w = initial.copy()
        t1 = t2 = None
        peak_error = 0.
        for step in range(c.max_steps+1):
            residual = x @ w-y
            train = np.mean(residual**2)
            pop = population(w-teacher,rho)
            if train<=c.epsilon and t1 is None:
                t1 = step
            if pop<=c.epsilon and t2 is None:
                t2 = step
            if step in samples:
                expected = [float(samples[step][k]) for k in ("train","population")]
                np.testing.assert_allclose([train,pop],expected,rtol=1e-9,atol=1e-12)
                peak_error = max(peak_error,np.max(np.abs(np.array(expected)-[train,pop])))
            if step >= max(samples) and None not in (t1,t2):
                break
            w -= c.step_size*gradient(x,residual,w,c.weight_decay)
        for key,value in [("t1",t1),("t2",t2)]:
            np.testing.assert_equal(target[key], "unobserved" if value is None else str(value))
        checks.append(dict(rho=rho,t1=t1,t2=t2,steps=step,max_abs_loss_error=peak_error))
    report = dict(isotropic_trials=len(baseline),literal_trial=1,checks=checks,seconds=perf_counter()-start)
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    main()

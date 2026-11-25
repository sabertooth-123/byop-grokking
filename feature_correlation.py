"""Paired equicorrelation experiment with exact discrete ridge GD dynamics."""
from dataclasses import asdict
import hashlib
import json
import os
import platform
from time import perf_counter

import numpy as np

from baseline import Config, ROOT, make_problem
from robustness import MISSING, RESAMPLES, TRIALS, trial_config
from weight_decay import plt, save_csv

RHOS = (0., .3, .6, .85)
BOOT_SEED = 4102026
OUT = ROOT / "results" / "feature_correlation"
CHECKPOINTS = np.unique(np.r_[0, 1, 10, 100, 1000, 5000, 10000,
                              np.geomspace(1, 30000, 61).round().astype(int)])


def correlate(base_x, rho):
    """Apply the symmetric square root of m Sigma to N(0,I/m) rows."""
    m = base_x.shape[1]
    bulk = np.sqrt(1-rho)
    spike = np.sqrt(1+(m-1)*rho)
    return bulk*base_x + (spike-bulk)*base_x.mean(axis=1, keepdims=True)


def population(error, rho):
    return ((1-rho)*(error @ error) + rho*error.sum()**2) / error.size


class Trajectory:
    """GD in the right singular basis, with an explicit training-null-space component."""
    def __init__(self, x, teacher, initial, c):
        n, self.m = x.shape
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        self.gram = s**2/n
        curvature = self.gram+c.weight_decay
        teacher_row = vt @ teacher
        initial_row = vt @ initial
        fixed = teacher_row*self.gram/curvature
        self.fixed_error = fixed-teacher_row
        self.start_error = initial_row-fixed
        self.q = 1-c.step_size*curvature
        self.q_null = 1-c.step_size*c.weight_decay
        if np.max(abs(self.q)) >= 1 or abs(self.q_null) >= 1:
            raise ValueError("Noncontracting GD dynamics for sampled training matrix")
        null_w = initial-vt.T @ initial_row
        null_y = teacher-vt.T @ teacher_row
        self.ww, self.wy, self.yy = null_w @ null_w, null_w @ null_y, null_y @ null_y
        self.axis_row = vt.sum(axis=1)/np.sqrt(self.m)
        self.axis_null_w = null_w.sum()/np.sqrt(self.m)
        self.axis_null_y = null_y.sum()/np.sqrt(self.m)

    def losses(self, steps, rho):
        z = self.q_null**steps
        row = self.fixed_error + self.start_error*self.q**steps[:, None]
        squared = row**2
        train = squared @ self.gram
        norm2 = squared.sum(axis=1) + z*z*self.ww - 2*z*self.wy + self.yy
        along = row @ self.axis_row + z*self.axis_null_w-self.axis_null_y
        spike = (rho+(1-rho)/self.m)*along**2
        bulk = (1-rho)/self.m*np.maximum(0., norm2-along**2)
        return train, spike+bulk, spike, bulk


def first_crossings(path, rho, epsilon, cap):
    t1 = t2 = None
    for start in range(0, cap+1, 1024):
        steps = np.arange(start, min(start+1024, cap+1))
        train, pop, _, _ = path.losses(steps, rho)
        for_train = np.flatnonzero(train <= epsilon)
        for_pop = np.flatnonzero(pop <= epsilon)
        if t1 is None and for_train.size:
            t1 = int(steps[for_train[0]])
        if t2 is None and for_pop.size:
            t2 = int(steps[for_pop[0]])
        if t1 is not None and t2 is not None:
            break
    return t1, t2


def spectrum(m, rho):
    low, high = (1-rho)/m, rho+(1-rho)/m
    return dict(bulk_eigenvalue=low, shared_axis_eigenvalue=high, condition_number=high/low,
                trace=high+(m-1)*low, participation_ratio=1/(high**2+(m-1)*low**2))


def summarize(rows):
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(TRIALS, size=(RESAMPLES, TRIALS))
    stats, alignment = [], []
    for rho in RHOS:
        group = sorted((r for r in rows if r["rho"] == rho), key=lambda r:r["trial"])
        if [r["trial"] for r in group] != list(range(1,TRIALS+1)):
            raise ValueError("Incomplete trial block")
        for key in ("t1","t2","delay","delta_t1","delta_t2","delta_delay","initial_train","initial_population"):
            vals = [r[key] for r in group if r[key] != MISSING]
            mean = sd = lo = hi = np.nan
            if len(vals) == TRIALS:
                a = np.array(vals)
                mean, sd = a.mean(), a.std(ddof=1)
                lo, hi = np.quantile(a[draws].mean(axis=1), [.025,.975])
            stats.append(dict(rho=rho, metric=key, observed=len(vals), total=TRIALS,
                              mean=mean, sd=sd, ci_low=lo, ci_high=hi))
        if rho:
            for metric in ("delta_t1","delta_t2","delta_delay"):
                x = np.array([r["teacher_axis_cos2"] for r in group])
                y = np.array([np.nan if r[metric]==MISSING else r[metric] for r in group])
                def corr(a,b):
                    a = a-a.mean(axis=-1, keepdims=True)
                    b = b-b.mean(axis=-1, keepdims=True)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        return (a*b).sum(axis=-1)/np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))
                lo, hi = np.quantile(corr(x[draws],y[draws]), [.025,.975])
                alignment.append(dict(rho=rho, metric=metric, pearson_r=corr(x,y), ci_low=lo, ci_high=hi))
    return stats, alignment


def plot(stats, curves):
    fig, axes = plt.subplots(1,2,figsize=(9,4),layout="constrained")
    for ax,key,title in zip(axes, ("delay","delta_delay"), ("Population grokking delay","Paired change from ρ = 0")):
        g = [r for r in stats if r["metric"]==key]
        y = np.array([r["mean"] for r in g])
        ax.errorbar(RHOS,y,yerr=[y-[r["ci_low"] for r in g],[r["ci_high"] for r in g]-y],
                    marker="o",capsize=4)
        ax.set(xlabel="Feature correlation ρ", ylabel="Gradient updates",title=title,xticks=RHOS)
    fig.suptitle("Trial means and paired bootstrap 95% intervals")
    fig.savefig(OUT/"delay.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,4),layout="constrained",sharey=True)
    for ax,key in zip(axes,("train","population")):
        for rho in RHOS:
            g = [r for r in curves if r["rho"]==rho]
            loss = np.array([r[key] for r in g]).reshape(TRIALS,-1)
            ax.plot(CHECKPOINTS,loss.mean(axis=0),label=f"ρ = {rho:g}")
        ax.set(xscale="symlog",yscale="log",xlabel="Gradient updates",title=key.capitalize()+" MSE")
        ax.axhline(.01,color=".5",ls="--")
    axes[0].set_ylabel("Mean across 20 trials")
    axes[1].legend()
    fig.savefig(OUT/"losses.png", dpi=180)
    plt.close(fig)


def main():
    start = perf_counter()
    base = Config.load()
    OUT.mkdir(exist_ok=True)
    rows, curves = [], []
    for trial in range(1,TRIALS+1):
        c = trial_config(base,trial)
        teacher,(z,_),_,initial = make_problem(c)
        for rho in RHOS:
            x = correlate(z,rho)
            path = Trajectory(x,teacher,initial,c)
            t1,t2 = first_crossings(path,rho,c.epsilon,c.max_steps)
            times = dict(t1=t1,t2=t2,delay=None if None in (t1,t2) else t2-t1)
            if rho==0:
                control = times.copy()
            row = dict(trial=trial,rho=rho,teacher_seed=c.teacher_seed,train_seed=c.train_seed,
                       student_seed=c.student_seed, teacher_axis_cos2=teacher.sum()**2/(teacher @ teacher)/c.dimension,
                       initial_train=np.mean((x @ (initial-teacher))**2),initial_population=population(initial-teacher,rho),
                       train_gram_min_positive=path.gram[-1], train_gram_max=path.gram[0], **spectrum(c.dimension,rho))
            for k,v in times.items():
                row[k] = MISSING if v is None else v
                row["delta_"+k] = MISSING if None in (v,control[k]) else v-control[k]
            rows.append(row)
            for step,tr,pop,spike,bulk in zip(CHECKPOINTS,*path.losses(CHECKPOINTS,rho)):
                curves.append(dict(trial=trial,rho=rho,step=step,train=tr,population=pop,
                                   shared_axis_loss=spike,orthogonal_loss=bulk))
    stats, alignment = summarize(rows)
    for name,data in [("trials",rows),("summary",stats),("losses",curves),("alignment",alignment)]:
        save_csv(OUT/(name+".csv"),data)
    plot(stats,curves)
    meta = dict(config=asdict(base), rhos=RHOS,trials=TRIALS,bootstrap_seed=BOOT_SEED,resamples=RESAMPLES,
                seed_stride=1000,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                openblas_threads=os.getenv("OPENBLAS_NUM_THREADS"),seconds=perf_counter()-start,
                source_sha256=hashlib.sha256((ROOT/"feature_correlation.py").read_bytes()).hexdigest())
    (OUT/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(meta)


if __name__=="__main__":
    main()

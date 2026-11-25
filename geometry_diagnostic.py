"""Milestone 4B: evaluate the same ridge iterates in two geometries."""
import csv
from dataclasses import asdict
import hashlib
import json
import os
import platform
from time import perf_counter

import numpy as np

from provenance import write_run
from baseline import Config, ROOT, make_problem
from feature_correlation import (CHECKPOINTS, MISSING, RESAMPLES, RHOS, TRIALS, Trajectory,
                                 correlate, first_crossings)
from robustness import trial_config
from weight_decay import plt, save_csv

OUT = ROOT / "results" / "geometry_diagnostic"
PARENT = ROOT / "results" / "feature_correlation"
ALIGNMENT_TRIALS = 6
SEED = 4202026
PARENT_COMMIT = "0620b2bbd87bba309a311af91ca032ed85f85486"


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def teachers(original):
    axis = np.ones_like(original)/np.sqrt(original.size)
    orthogonal = original-axis*(original @ axis)
    return dict(random=original, parallel=axis, orthogonal=orthogonal/np.linalg.norm(orthogonal))


def difference(a, b):
    return MISSING if MISSING in (a, b) else a-b


def evaluate(x, target, initial, c, rho):
    path = Trajectory(x, target, initial, c)
    # Training geometry is fixed in path. The argument to losses selects only the scoring covariance.
    train, pop = first_crossings(path, rho, c.epsilon, c.max_steps)
    _, iso = first_crossings(path, 0., c.epsilon, c.max_steps) if rho else (train, pop)
    row = {k: MISSING if v is None else v for k,v in (("t1",train),("t_pop",pop),("t_iso",iso))}
    for metric in ("pop","iso"):
        row["delay_"+metric] = difference(row["t_"+metric],row["t1"])
    row["metric_gap"] = difference(row["t_pop"],row["t_iso"])
    _, pop_loss, _, _ = path.losses(CHECKPOINTS,rho)
    _, iso_loss, _, _ = path.losses(CHECKPOINTS,0.)
    return row, pop_loss, iso_loss


def summarize(rows):
    stats = []
    for kind in ("random","parallel","orthogonal"):
        size = TRIALS if kind=="random" else ALIGNMENT_TRIALS
        draws = np.random.default_rng(SEED).integers(size,size=(RESAMPLES,size))
        for rho in RHOS:
            group = sorted((r for r in rows if r["teacher"]==kind and r["rho"]==rho),key=lambda r:r["trial"])
            assert [r["trial"] for r in group]==list(range(1,size+1))
            metrics = ["t1","t_pop","t_iso","delay_pop","delay_iso","metric_gap",
                       "delta_t1","delta_t_pop","delta_t_iso","delta_delay_pop","delta_delay_iso"]
            if kind=="parallel":
                metrics += ["parallel_minus_orthogonal_"+k for k in ("t1","t_pop","t_iso")]
            for key in metrics:
                vals = [r[key] for r in group if r[key]!=MISSING]
                mean = sd = low = high = np.nan
                if len(vals)==size:
                    a = np.array(vals)
                    mean,sd = a.mean(),a.std(ddof=1)
                    low,high = np.quantile(a[draws].mean(axis=1),[.025,.975])
                stats.append(dict(teacher=kind,rho=rho,metric=key,trials=size,observed=len(vals),
                                  mean=mean,sd=sd,ci_low=low,ci_high=high))
    return stats


def plots(stats, curves):
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharey=True,layout="constrained")
    for ax,kind in zip(axes,("random","parallel","orthogonal")):
        for metric,label,marker in [("t_pop","Training distribution","o"),("t_iso","Common isotropic","x")]:
            g = [s for s in stats if s["teacher"]==kind and s["metric"]==metric]
            y = np.array([s["mean"] for s in g])
            ax.errorbar(RHOS,y,yerr=[y-[s["ci_low"] for s in g],[s["ci_high"] for s in g]-y],
                        marker=marker,label=label,capsize=3)
        ax.set(title=f"{kind.capitalize()} teacher",xlabel="Training correlation ρ",xticks=RHOS)
    axes[0].set_ylabel("Updates to MSE ≤ 0.01")
    axes[-1].legend()
    fig.suptitle("First crossings · means and pointwise 95% intervals")
    fig.savefig(OUT/"crossings.png",dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(9,4),sharey=True,layout="constrained")
    for ax,key,title in zip(axes,("normalized_pop","normalized_iso"),("Training distribution","Common isotropic")):
        for rho in RHOS:
            selected = [r[key] for r in curves if r["teacher"]=="random" and r["rho"]==rho]
            y = np.reshape(selected,(TRIALS,-1)).mean(axis=0)
            ax.plot(CHECKPOINTS,y,label=f"ρ = {rho:g}")
        ax.set(xscale="symlog",yscale="log",title=title,xlabel="Gradient updates")
    axes[0].set_ylabel("Mean of trial loss / initial trial loss")
    axes[1].legend()
    fig.savefig(OUT/"normalized_losses.png",dpi=180)
    plt.close(fig)


def main():
    start = perf_counter()
    base = Config.load()
    OUT.mkdir(exist_ok=True)
    original = {(int(r["trial"]),float(r["rho"])):r for r in read_rows(PARENT/"trials.csv")}
    rows,curves = [],[]
    for trial in range(1,TRIALS+1):
        c = trial_config(base,trial)
        teacher,(z,_),_,w = make_problem(c)
        control = {}
        for rho in RHOS:
            x = correlate(z,rho)
            group = {}
            for kind,target in teachers(teacher).items():
                if kind!="random" and trial>ALIGNMENT_TRIALS:
                    continue
                counts,pop,iso = evaluate(x,target,w,c,rho)
                if rho==0:
                    control[kind] = counts.copy()
                keys = dict(trial=trial,teacher=kind,rho=rho)
                row = dict(**keys,teacher_seed=c.teacher_seed,train_seed=c.train_seed,student_seed=c.student_seed,
                           teacher_norm=np.linalg.norm(target),squared_axis_projection=target.sum()**2/c.dimension,
                           **counts)
                for k in ("t1","t_pop","t_iso","delay_pop","delay_iso"):
                    row["delta_"+k] = difference(counts[k],control[kind][k])
                for k in ("t1","t_pop","t_iso"):
                    row["parallel_minus_orthogonal_"+k] = "NA"
                if kind=="random":
                    saved = original[trial,rho]
                    for k,ref in [("t1","t1"),("t_pop","t2"),("delay_pop","delay")]:
                        np.testing.assert_equal(str(counts[k]),saved[ref])
                rows.append(row)
                group[kind] = row
                for step,p,i in zip(CHECKPOINTS,pop,iso):
                    curves.append(dict(**keys,step=step,population=p,isotropic=i,
                                       normalized_pop=p/pop[0],normalized_iso=i/iso[0]))
            if "parallel" in group:
                for k in ("t1","t_pop","t_iso"):
                    group["parallel"]["parallel_minus_orthogonal_"+k] = difference(group["parallel"][k],group["orthogonal"][k])
    stats = summarize(rows)
    for file,data in [("trials.csv",rows),("checkpoints.csv",curves),("summary.csv",stats)]:
        save_csv(OUT/file,data)
    plots(stats,curves)
    meta = dict(parent_experiment_commit=PARENT_COMMIT,config=asdict(base),rhos=RHOS,random_trials=TRIALS,
                alignment_trials=ALIGNMENT_TRIALS,seed_stride=1000,bootstrap_seed=SEED,resamples=RESAMPLES,
                python=platform.python_version(),numpy=np.__version__,openblas_threads=os.getenv("OPENBLAS_NUM_THREADS"),
                source_sha256=hashlib.sha256((ROOT/"geometry_diagnostic.py").read_bytes()).hexdigest(),
                seconds=perf_counter()-start)
    write_run(OUT/"run.json", meta,
              "Milestone 4 paired seeds, stride 1000; first alignment_trials seeds for both unit teacher controls.",
              "Replay equicorrelated Gaussian designs; score the same iterates under training and isotropic covariances.")
    print(meta)


if __name__=="__main__":
    main()

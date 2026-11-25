"""Audit the reviewed research archive without running training or editing results."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from setup_reference import REVISION

ROOT = Path(__file__).resolve().parent
REVIEWED = "7e9a62052a93037f6f0ba72a4502abfa111e4ff8"
PARENT = "0620b2bbd87bba309a311af91ca032ed85f85486"


def git(*args):
    return subprocess.check_output(["git","-C",str(ROOT),*args])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_hash(name, digest):
    # Historical Windows working files may have had CRLF, including mixed endings.
    blob = git("show", f"{REVIEWED}:{name}")
    candidates = [blob, blob.replace(b"\n",b"\r\n")]
    require(any(hashlib.sha256(b).hexdigest()==digest for b in candidates),
            f"Historical source digest mismatch: {name}")


def main():
    archive = ROOT/"results"
    listing = git("ls-tree","-r","--name-only",REVIEWED,"--","results").decode().splitlines()
    actual = {p.relative_to(ROOT).as_posix() for p in archive.rglob("*") if p.is_file()}
    require(set(listing)==actual, "Unexpected or missing archive files")
    # Git applies this checkout's line-ending rules for text and compares binaries exactly.
    git("diff","--exit-code",REVIEWED,"--","results")
    for directory in [archive,*sorted(p for p in archive.iterdir() if p.is_dir())]:
        run = json.loads((directory/"run.json").read_text())
        cfg = run.get("base_config",run.get("config"))
        require(cfg==json.loads(git("show",f"{REVIEWED}:baseline.json")), "Baseline configuration mismatch")
        receipt = directory/("comparison.json" if directory==archive else "verification.json")
        proof = json.loads(receipt.read_text())
        if directory.name=="geometry_diagnostic":
            require(run["reference_commit"]==PARENT, "Unexpected local parent commit")
        else:
            for record in [run,proof]:
                if "reference_commit" in record:
                    require(record["reference_commit"]==REVISION, "Unexpected authors' revision")
        hashes = run.get("source_sha256",run.get("sha256",{}))
        if isinstance(hashes,str):
            hashes = {directory.name+".py":hashes}
        for name,digest in hashes.items():
            check_hash(name,digest)
        if "verifier_sha256" in proof:
            check_hash("verify_figure2bcd.py",proof["verifier_sha256"])
        if (directory/"trials.csv").exists():
            trials = rows(directory/"trials.csv")
            for r in trials:
                for t2,delay in ([("t_pop","delay_pop"),("t_iso","delay_iso")]
                                 if "t_pop" in r else [("t2","delay")]):
                    expected = ("unobserved" if "unobserved" in (r["t1"],r[t2])
                                else str(int(r[t2])-int(r["t1"])))
                    require(r[delay]==expected, f"Incorrect paired delay in {directory.name}")
        if directory==archive:
            require(run["observations"]==proof["observations"]==len(rows(directory/"losses.csv")),
                    "Baseline history length mismatch")
        elif directory.name=="robustness":
            for name,count in [("threshold_trials",4),("population_trials",5)]:
                groups = Counter(int(r["trial"]) for r in rows(directory/(name+".csv")))
                require(groups==dict.fromkeys(range(1,21),count),f"Unbalanced {name}")
        print(f"OK  {directory.relative_to(ROOT)}")
    print(f"Validated {len(listing)} reviewed artifacts, recorded source hashes and numerical records.")


if __name__=="__main__":
    main()

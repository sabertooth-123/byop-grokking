"""Record execution provenance for new runs. Archived receipts are read separately."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
import numpy as np

from setup_reference import REVISION

ROOT = Path(__file__).resolve().parent


def execution():
    sources = {ROOT/"baseline.json", ROOT/"requirements.txt"}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and str(file).endswith(".py"):
            path = Path(file).resolve()
            if path.parent in (ROOT, ROOT/"reference"):
                sources.add(path)
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(sources)}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return dict(project_head=git("rev-parse","HEAD"),
                working_tree_source_changes=git("status","--porcelain","--",*hashes).splitlines(),
                source_sha256=hashes, authors_reference_pin=REVISION,
                python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__)


def write_run(path, metadata, seed_protocol, dataset):
    metadata.update(execution=execution(), seed_protocol=seed_protocol, dataset=dataset,
                    validation={"status":"pending"})
    path.write_text(json.dumps(metadata,indent=2)+"\n")


def write_validation(path, report):
    run_path = path.parent/"run.json"
    run = json.loads(run_path.read_text())
    report["verification_execution"] = execution()
    path.write_text(json.dumps(report,indent=2)+"\n")
    if "execution" in run:
        run["validation"] = dict(status="passed", receipt=path.name,
                                 receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        run_path.write_text(json.dumps(run,indent=2)+"\n")

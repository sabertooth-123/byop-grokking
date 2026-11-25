"""Prepare the pinned upstream checkout used by compare_reference.py."""
from pathlib import Path
import subprocess
import tempfile

REFERENCE = Path(__file__).resolve().parent / "reference"
URL = "https://github.com/raymondmyxu/to-grok-grokking.git"
REVISION = "4b01ed1d878a37a36b7872e6cbd2beabb8b23cce"


def verify_checkout(path):
    if not (path / ".git").exists():
        raise ValueError(f"Expected a Git checkout at {path}. Run python setup_reference.py.")

    def git_output(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    if git_output("rev-parse", "HEAD") != REVISION:
        raise ValueError(f"Expected {REVISION} at {path}. Preserve any local work and "
                         "move the checkout aside before rerunning python setup_reference.py.")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Reference checkout has edited tracked files at {path}. "
                         "Preserve your edits and restore a clean checkout.")


def main():
    if not REFERENCE.exists():
        with tempfile.TemporaryDirectory(dir=REFERENCE.parent) as temp:
            clone = Path(temp) / "checkout"
            subprocess.run(["git", "clone", "--no-checkout", URL, str(clone)], check=True)
            subprocess.run(["git", "-C", str(clone), "checkout", "--detach", REVISION], check=True)
            verify_checkout(clone)
            clone.rename(REFERENCE)
    verify_checkout(REFERENCE)
    print(f"Reference ready at {REFERENCE} ({REVISION})")


if __name__ == "__main__":
    main()

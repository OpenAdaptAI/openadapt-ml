"""Kaggle kernel entrypoint for openadapt-ml SFT training.

Pushed by ``openadapt_ml.cloud.kaggle_cloud``. It reads the staged dataset
(bundle, config, optional checkpoint) from /kaggle/input, trains on the
session GPU, and writes results to /kaggle/working/results so
``kaggle kernels output`` can pull them back.

It exits with code 3 before installing anything if the session has no GPU or
no internet: a run that falls back to CPU must not pass for a GPU run.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PIP_SPEC = os.environ.get("OPENADAPT_ML_PIP_SPEC", "openadapt-ml[training]")
WORK = Path("/kaggle/working")
RESULTS = WORK / "results"


def _find_staged() -> Path:
    hits = sorted(Path("/kaggle/input").rglob("openadapt_kaggle_stage.json"))
    if not hits:
        raise SystemExit("staged dataset not found under /kaggle/input")
    return hits[0].parent


def _preflight() -> None:
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    print("GPU", probe.stdout.strip() or probe.stderr.strip(), flush=True)
    if probe.returncode != 0 or "GPU" not in probe.stdout:
        print("No GPU in this session; refusing to train on CPU.", flush=True)
        raise SystemExit(3) from None
    try:
        socket.gethostbyname("pypi.org")
    except OSError as exc:
        print(f"No internet in this session ({exc}).", flush=True)
        raise SystemExit(3) from None


def main() -> int:
    _preflight()
    staged = _find_staged()
    stage = json.loads((staged / "openadapt_kaggle_stage.json").read_text())
    print("stage", stage, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", PIP_SPEC], check=True)

    bundle = WORK / "bundle"
    if not bundle.exists():
        shutil.copytree(staged / "bundle", bundle)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (staged / "checkpoint").is_dir():
        shutil.copytree(
            staged / "checkpoint", RESULTS / "checkpoint_in", dirs_exist_ok=True
        )

    cmd = [
        sys.executable,
        "-m",
        "openadapt_ml.scripts.train",
        "--config",
        str(staged / "train_config.yaml"),
        "--jsonl",
        str(bundle / "training_data.jsonl"),
        "--output-dir",
        str(RESULTS),
    ]
    print("Running:", " ".join(cmd), flush=True)
    started = time.time()
    with open(RESULTS / "train_stdout.log", "w") as log:
        rc = subprocess.run(
            cmd, cwd=bundle, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    summary = {
        "status": "completed" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_time": time.time() - started,
        "stage": stage,
    }
    (RESULTS / "kaggle_run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

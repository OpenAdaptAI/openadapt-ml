"""Kaggle GPU training for OpenAdapt (free T4/P100 sessions).

Kaggle replaces Modal for GPU training. The flow uses only the ``kaggle`` CLI:

1. ``stage``: builds a private dataset folder (bundle, config, optional
   checkpoint) and a script-kernel folder whose ``kernel-metadata.json`` turns
   on GPU and internet.
2. ``push``: ``kaggle datasets create`` (or ``version``), then
   ``kaggle kernels push``. This starts a free Kaggle session; it never
   touches Modal.
3. ``status`` / ``pull``: ``kaggle kernels status`` and
   ``kaggle kernels output`` bring the results back.

To resume, pass the pulled ``results`` directory as ``--checkpoint`` to the
next ``stage``; it is uploaded with the dataset as ``checkpoint/``.

Credentials: ``~/.kaggle/access_token`` or ``~/.kaggle/kaggle.json``, or the
``KAGGLE_USERNAME``/``KAGGLE_KEY`` (or ``KAGGLE_API_TOKEN``) environment
variables. GPU and internet in kernels need a phone-verified Kaggle account.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

KERNEL_FILE = "kaggle_train_kernel.py"
STAGE_MARKER = "openadapt_kaggle_stage.json"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{4,48}[a-z0-9]$")
ACCELERATORS = ("NvidiaTeslaT4", "NvidiaTeslaP100")


def _check_slug(value: str, what: str) -> str:
    if not SLUG_RE.fullmatch(value):
        raise ValueError(
            f"{what} must be 6-50 lowercase letters, digits or dashes: {value!r}"
        )
    return value


def kernel_metadata(
    username: str, kernel_slug: str, dataset_slug: str, accelerator: str
) -> dict:
    """Kaggle kernel-metadata.json for a private GPU script kernel."""
    if accelerator not in ACCELERATORS:
        raise ValueError(f"accelerator must be one of {ACCELERATORS}")
    return {
        "id": f"{username}/{_check_slug(kernel_slug, 'kernel slug')}",
        "title": kernel_slug,
        "code_file": KERNEL_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [f"{username}/{_check_slug(dataset_slug, 'dataset slug')}"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def dataset_metadata(username: str, dataset_slug: str) -> dict:
    return {
        "title": dataset_slug,
        "id": f"{username}/{_check_slug(dataset_slug, 'dataset slug')}",
        "licenses": [{"name": "other"}],
    }


def stage(
    bundle: Path,
    config: Path,
    out: Path,
    username: str,
    dataset_slug: str,
    kernel_slug: str,
    accelerator: str = "NvidiaTeslaT4",
    checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    """Write <out>/dataset and <out>/kernel. Returns both paths."""
    if not (bundle / "training_data.jsonl").is_file():
        raise FileNotFoundError(f"{bundle}/training_data.jsonl not found")
    if not config.is_file():
        raise FileNotFoundError(config)
    dataset_dir, kernel_dir = out / "dataset", out / "kernel"
    for d in (dataset_dir, kernel_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    shutil.copytree(bundle, dataset_dir / "bundle")
    shutil.copy2(config, dataset_dir / "train_config.yaml")
    if checkpoint is not None:
        shutil.copytree(checkpoint, dataset_dir / "checkpoint")
    (dataset_dir / STAGE_MARKER).write_text(
        json.dumps(
            {
                "dataset": dataset_slug,
                "kernel": kernel_slug,
                "has_checkpoint": checkpoint is not None,
            }
        )
    )
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata(username, dataset_slug), indent=2)
    )
    shutil.copy2(Path(__file__).with_name(KERNEL_FILE), kernel_dir / KERNEL_FILE)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            kernel_metadata(username, kernel_slug, dataset_slug, accelerator), indent=2
        )
    )
    return dataset_dir, kernel_dir


def _kaggle(*args: str) -> int:
    print("+ kaggle", " ".join(args), flush=True)
    return subprocess.run(["kaggle", *args]).returncode


def push(staged: Path, new_dataset: bool, message: str) -> int:
    dataset_dir, kernel_dir = staged / "dataset", staged / "kernel"
    if new_dataset:
        rc = _kaggle("datasets", "create", "-p", str(dataset_dir), "-r", "zip")
    else:
        rc = _kaggle(
            "datasets", "version", "-p", str(dataset_dir), "-m", message, "-r", "zip"
        )
    if rc:
        return rc
    print(
        "Wait until the dataset version is ready before the kernel reads it.",
        flush=True,
    )
    return _kaggle("kernels", "push", "-p", str(kernel_dir))


def _kernel_ref(staged: Path) -> str:
    return json.loads((staged / "kernel" / "kernel-metadata.json").read_text())["id"]


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kaggle GPU training for OpenAdapt")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "stage", help="Write the dataset and kernel folders (no network)"
    )
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("training_output/kaggle"))
    s.add_argument("--username", required=True, help="Kaggle username")
    s.add_argument("--dataset-slug", default="openadapt-ml-train-data")
    s.add_argument("--kernel-slug", default="openadapt-ml-train")
    s.add_argument("--accelerator", default="NvidiaTeslaT4", choices=ACCELERATORS)
    s.add_argument("--checkpoint", type=Path, help="Pulled results dir to resume from")
    p = sub.add_parser("push", help="Upload the dataset and start the kernel")
    p.add_argument("--staged", type=Path, default=Path("training_output/kaggle"))
    p.add_argument(
        "--new-dataset", action="store_true", help="First upload (datasets create)"
    )
    p.add_argument("--message", default="openadapt-ml training data")
    st = sub.add_parser("status", help="Kernel status")
    st.add_argument("--staged", type=Path, default=Path("training_output/kaggle"))
    pl = sub.add_parser("pull", help="Download kernel output (results)")
    pl.add_argument("--staged", type=Path, default=Path("training_output/kaggle"))
    pl.add_argument("--out", type=Path, default=Path("training_output/kaggle/output"))
    args = parser.parse_args(argv)

    if args.command == "stage":
        d, k = stage(
            args.bundle,
            args.config,
            args.out,
            args.username,
            args.dataset_slug,
            args.kernel_slug,
            args.accelerator,
            args.checkpoint,
        )
        print(f"dataset: {d}\nkernel:  {k}")
        return 0
    if args.command == "push":
        return push(args.staged, args.new_dataset, args.message)
    if args.command == "status":
        return _kaggle("kernels", "status", _kernel_ref(args.staged))
    args.out.mkdir(parents=True, exist_ok=True)
    return _kaggle("kernels", "output", _kernel_ref(args.staged), "-p", str(args.out))


if __name__ == "__main__":
    sys.exit(cli_main())

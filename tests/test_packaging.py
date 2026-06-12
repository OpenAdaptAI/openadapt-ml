"""Built-artifact checks.

CI normally tests the repo checkout, where files like configs/ exist
regardless of packaging configuration. OpenAdaptAI/OpenAdapt#999 bug 5
shipped because the wheel silently lacked the configs/ directory and
nothing ever inspected a built artifact. These tests build the wheel
and assert its contents.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "could not build wheel (is 'build' installed?): " + result.stderr[-500:]
        )
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_wheel_contains_bundled_configs(wheel_path: Path):
    with zipfile.ZipFile(wheel_path) as zf:
        config_yamls = [
            n
            for n in zf.namelist()
            if n.startswith("openadapt_ml/configs/") and n.endswith(".yaml")
        ]
    assert len(config_yamls) >= 5, (
        "Wheel is missing bundled training configs; cmd_train and the "
        "openadapt CLI resolve default configs from openadapt_ml/configs/ "
        f"inside the installed package. Found: {config_yamls}"
    )


def test_wheel_contains_core_subpackages(wheel_path: Path):
    required = {
        "openadapt_ml/__init__.py",
        "openadapt_ml/cloud/local.py",
        "openadapt_ml/scripts/train.py",
        "openadapt_ml/training/trainer.py",
        "openadapt_ml/ingest/capture.py",
        "openadapt_ml/schema/__init__.py",
    }
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    missing = required - names
    assert not missing, f"Wheel is missing core modules: {sorted(missing)}"


def test_wheel_version_matches_pyproject(wheel_path: Path):
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert match, "could not find version in pyproject.toml"
    assert f"-{match.group(1)}-" in wheel_path.name, (
        f"wheel {wheel_path.name} does not match pyproject version {match.group(1)}"
    )

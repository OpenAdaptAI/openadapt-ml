"""Kaggle staging and the Modal GPU production guard (no network)."""

from __future__ import annotations

import json

import pytest

from openadapt_ml.cloud import kaggle_cloud
from openadapt_ml.cloud.modal_gpu_guard import modal_gpu_refusal


def test_guard_refuses_by_default():
    assert "off by default" in modal_gpu_refusal({}, profile="other")


@pytest.mark.parametrize("environment", ["", "main", "prod", "Production"])
def test_guard_refuses_production_environments(environment):
    env = {"OPENADAPT_ALLOW_MODAL_GPU": "1", "MODAL_ENVIRONMENT": environment}
    assert "non-production" in modal_gpu_refusal(env, profile="research")


def test_guard_refuses_production_workspace():
    env = {"OPENADAPT_ALLOW_MODAL_GPU": "1", "MODAL_ENVIRONMENT": "gpu-research"}
    assert "production workspace" in modal_gpu_refusal(env, profile="abrichr")
    assert modal_gpu_refusal(env, profile="") is not None


def test_guard_allows_explicit_nonprod_target():
    env = {"OPENADAPT_ALLOW_MODAL_GPU": "1", "MODAL_ENVIRONMENT": "gpu-research"}
    assert modal_gpu_refusal(env, profile="openadapt-research") is None


def test_modal_train_cli_refuses_without_opt_in(monkeypatch, tmp_path, capsys):
    from openadapt_ml.cloud.modal_cloud import cli_main

    monkeypatch.delenv("OPENADAPT_ALLOW_MODAL_GPU", raising=False)
    rc = cli_main(["train", "--bundle", str(tmp_path)])
    assert rc == 1
    assert "Refusing Modal GPU work" in capsys.readouterr().out


def test_stage_writes_gpu_kernel_and_dataset(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "training_data.jsonl").write_text("{}\n")
    config = tmp_path / "c.yaml"
    config.write_text("model: x\n")
    ckpt = tmp_path / "results"
    ckpt.mkdir()
    (ckpt / "adapter.bin").write_bytes(b"0")
    d, k = kaggle_cloud.stage(
        bundle,
        config,
        tmp_path / "out",
        "someone",
        "oa-train-data",
        "oa-train",
        checkpoint=ckpt,
    )
    meta = json.loads((k / "kernel-metadata.json").read_text())
    assert (
        meta["enable_gpu"] is True
        and meta["enable_internet"] is True
        and meta["is_private"] is True
    )
    assert meta["dataset_sources"] == ["someone/oa-train-data"]
    assert (k / meta["code_file"]).is_file()
    assert (d / "bundle" / "training_data.jsonl").is_file()
    assert (d / "checkpoint" / "adapter.bin").is_file()
    assert (
        json.loads((d / "dataset-metadata.json").read_text())["id"]
        == "someone/oa-train-data"
    )


def test_stage_rejects_bad_slug_and_missing_bundle(tmp_path):
    with pytest.raises(ValueError):
        kaggle_cloud.kernel_metadata("u", "Bad Slug", "oa-train-data", "NvidiaTeslaT4")
    with pytest.raises(FileNotFoundError):
        kaggle_cloud.stage(
            tmp_path,
            tmp_path / "c.yaml",
            tmp_path / "o",
            "u",
            "oa-train-data",
            "oa-train",
        )

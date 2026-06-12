import json
import importlib.util
import sys

from dreamaze.huggingface_jobs import (
    BEST_GPU_HARDWARE_FLAVOR,
    HuggingFaceJobConfig,
    build_huggingface_job_command,
    load_huggingface_job_config,
)
from dreamaze.huggingface_jobs_cli import run_huggingface_jobs_cli


def test_huggingface_job_config_builds_remote_ready_dry_run_command(tmp_path):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "hardware_flavor": "t4-small",
                "timeout": "45m",
                "python": "3.12",
                "repo": "Srini410/dreamaze-job-runs",
                "namespace": "Srini410",
                "dataset_preset": "tiny",
                "dataset_dir": "/tmp/dreamaze/dataset",
                "checkpoint_dir": "/tmp/dreamaze/checkpoints",
                "evaluation_output": "/tmp/dreamaze/evaluation.json",
                "batch_size": 4,
                "sampling_steps": 6,
                "max_train_steps": 7,
                "checkpoint_every_steps": 2,
                "learning_rate": 0.025,
                "training_seed": 123,
                "evaluation_seed": 456,
                "retry_count": 3,
                "eval_split": "test",
                "device": "cuda",
                "precision": "float16",
                "num_workers": 2,
                "output_repo": "Srini410/dreamaze-artifacts",
                "output_repo_type": "dataset",
                "output_path_in_repo": "runs/smoke",
                "env": {"HF_XET_HIGH_PERFORMANCE": "1"},
            }
        )
    )

    config = load_huggingface_job_config(config_path)
    command = build_huggingface_job_command(config)

    assert config == HuggingFaceJobConfig(
        hardware_flavor="t4-small",
        timeout="45m",
        python="3.12",
        repo="Srini410/dreamaze-job-runs",
        namespace="Srini410",
        dataset_preset="tiny",
        dataset_dir="/tmp/dreamaze/dataset",
        checkpoint_dir="/tmp/dreamaze/checkpoints",
        evaluation_output="/tmp/dreamaze/evaluation.json",
        batch_size=4,
        sampling_steps=6,
        max_train_steps=7,
        checkpoint_every_steps=2,
        learning_rate=0.025,
        training_seed=123,
        evaluation_seed=456,
        retry_count=3,
        eval_split="test",
        device="cuda",
        precision="float16",
        num_workers=2,
        output_repo="Srini410/dreamaze-artifacts",
        output_repo_type="dataset",
        output_path_in_repo="runs/smoke",
        env={"HF_XET_HIGH_PERFORMANCE": "1"},
    )
    assert command[:4] == ["hf", "jobs", "uv", "run"]
    assert command.index("--flavor") < command.index("jobs/dreamaze_hf_job.py")
    assert command.index("--timeout") < command.index("jobs/dreamaze_hf_job.py")
    assert "--flavor" in command
    assert "t4-small" in command
    assert "--timeout" in command
    assert "45m" in command
    assert "--secrets" in command
    assert "HF_TOKEN" in command
    assert "--batch-size" in command
    assert "4" in command
    assert "--sampling-steps" in command
    assert "6" in command
    assert "--checkpoint-every-steps" in command
    assert "2" in command
    assert "--output-repo" in command
    assert "Srini410/dreamaze-artifacts" in command


def test_huggingface_job_config_can_select_best_gpu_profile(tmp_path):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "compute_profile": "best_gpu",
                "output_repo": "Srini410/dreamaze-artifacts",
            }
        )
    )

    config = load_huggingface_job_config(config_path)
    command = build_huggingface_job_command(config)

    assert config.hardware_flavor == BEST_GPU_HARDWARE_FLAVOR
    assert config.dataset_preset == "first"
    assert config.device == "cuda"
    assert config.precision == "float16"
    assert config.batch_size == 64
    assert config.sampling_steps == 64
    assert config.max_train_steps == 5000
    assert config.checkpoint_every_steps == 250
    assert config.timeout == "8h"
    assert "--flavor" in command
    assert BEST_GPU_HARDWARE_FLAVOR in command
    assert "--device" in command
    assert "cuda" in command


def test_huggingface_job_cli_dry_run_prints_command_without_launching(tmp_path, capsys):
    config_path = tmp_path / "hf-job.json"
    config_path.write_text(
        json.dumps(
            {
                "hardware_flavor": "cpu-basic",
                "timeout": "10m",
                "dataset_preset": "tiny",
                "batch_size": 1,
                "sampling_steps": 2,
                "max_train_steps": 1,
                "checkpoint_every_steps": 1,
            }
        )
    )

    exit_code = run_huggingface_jobs_cli(["--config", str(config_path), "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Hugging Face Jobs dry run" in output
    assert "hf jobs uv run" in output
    assert "--flavor cpu-basic" in output
    assert "--dataset-preset tiny" in output


def test_huggingface_job_script_runs_tiny_build_train_evaluate_workflow(
    tmp_path, monkeypatch
):
    script_path = "jobs/dreamaze_hf_job.py"
    spec = importlib.util.spec_from_file_location("dreamaze_hf_job", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    evaluation_output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script_path,
            "--dataset-preset",
            "tiny",
            "--dataset-dir",
            str(dataset_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--evaluation-output",
            str(evaluation_output),
            "--batch-size",
            "1",
            "--sampling-steps",
            "2",
            "--max-train-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
            "--learning-rate",
            "0.01",
            "--training-seed",
            "7",
            "--evaluation-seed",
            "11",
            "--retry-count",
            "1",
            "--eval-split",
            "validation",
            "--device",
            "cpu",
            "--precision",
            "float32",
            "--num-workers",
            "0",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert (dataset_dir / "manifest.json").exists()
    assert (checkpoint_dir / "checkpoint-step-000001.json").exists()
    report = json.loads(evaluation_output.read_text())
    assert report["dataset_split"] == "validation"
    assert report["official_score"] == "single_sample_success"

import json
from dataclasses import replace

from dreamaze.dataset import build_training_example, load_dataset_artifact_shard
from dreamaze.dataset_cli import (
    first_dataset_size_config,
    larger_dataset_config,
    run_dataset_builder_cli,
)


def test_tiny_dataset_builder_cli_writes_artifacts_manifest_and_previews(
    tmp_path, capsys
):
    exit_code = run_dataset_builder_cli(
        [
            "build",
            "--preset",
            "tiny",
            "--output-dir",
            str(tmp_path),
            "--preview-images",
        ]
    )

    assert exit_code == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["split_counts"] == {"test": 1, "train": 3, "validation": 1}
    assert [shard["name"] for shard in manifest["shards"]] == [
        "train-00000.npz",
        "train-00001.npz",
        "validation-00000.npz",
        "test-00000.npz",
    ]
    assert load_dataset_artifact_shard(tmp_path / "train-00000.npz")["seed"] == [
        10,
        11,
    ]
    assert [path.name for path in sorted((tmp_path / "previews").glob("*.pgm"))] == [
        "test-000200.pgm",
        "train-000010.pgm",
        "train-000011.pgm",
        "train-000012.pgm",
        "validation-000100.pgm",
    ]

    report = capsys.readouterr().out
    assert "Dataset Builder preset: tiny" in report
    assert "Manifest status: complete" in report
    assert "Output directory:" in report
    assert "Generated counts: train=3 validation=1 test=1" in report


def test_tiny_dataset_builder_cli_reports_shards_and_resumes_completed_artifacts(
    tmp_path, capsys
):
    first_exit_code = run_dataset_builder_cli(
        ["build", "--preset", "tiny", "--output-dir", str(tmp_path)]
    )
    first_manifest = json.loads((tmp_path / "manifest.json").read_text())

    second_exit_code = run_dataset_builder_cli(
        ["build", "--preset", "tiny", "--output-dir", str(tmp_path)]
    )
    second_manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert second_manifest == first_manifest

    report = capsys.readouterr().out
    assert "Expected rejections: train=0 validation=0 test=0" in report
    assert "Invariant failures: 0" in report
    assert "Shard progress: 4/4 complete" in report
    assert "Resume: reused completed Dataset Artifacts" in report


def test_dataset_builder_cli_fails_loudly_on_invariant_failure(tmp_path, capsys):
    def build_invalid_training_example(*, seed, config):
        example = build_training_example(seed=seed, config=config)
        empty_label = tuple(
            tuple(False for _ in row) for row in example.training_label
        )
        return replace(example, training_label=empty_label)

    exit_code = run_dataset_builder_cli(
        ["build", "--preset", "tiny", "--output-dir", str(tmp_path)],
        training_example_builder=build_invalid_training_example,
    )

    assert exit_code == 1
    assert not (tmp_path / "manifest.json").exists()
    assert "Dataset Builder failed:" in capsys.readouterr().err


def test_first_dataset_size_config_uses_published_split_counts():
    config = first_dataset_size_config()

    assert config.split_sizes == {
        "train": 10_000,
        "validation": 1_000,
        "test": 1_000,
    }


def test_larger_dataset_config_uses_bounded_training_run_counts():
    config = larger_dataset_config()

    assert config.split_sizes == {
        "train": 50_000,
        "validation": 5_000,
        "test": 5_000,
    }
    assert config.shard_size == 2048

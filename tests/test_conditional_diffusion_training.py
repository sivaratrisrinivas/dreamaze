import json

from dreamaze.dataset import (
    DatasetConfig,
    DatasetSplitName,
    MazeFamily,
    write_dataset_artifacts,
)
from dreamaze.training import (
    TrainingConfig,
    load_training_config,
    train_conditional_diffusion_solver,
)
from dreamaze.training_cli import run_training_cli


def test_tiny_dataset_artifacts_flow_through_training_and_write_checkpoint(tmp_path):
    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    write_dataset_artifacts(
        config=DatasetConfig(
            width=4,
            height=4,
            maze_families=(MazeFamily.KRUSKAL, MazeFamily.WILSON),
            split_sizes={
                DatasetSplitName.TRAIN: 2,
                DatasetSplitName.VALIDATION: 0,
                DatasetSplitName.TEST: 0,
            },
            seed_ranges={
                DatasetSplitName.TRAIN: range(10, 20),
                DatasetSplitName.VALIDATION: range(100, 101),
                DatasetSplitName.TEST: range(200, 201),
            },
            shard_size=1,
            minimum_path_length=2,
        ),
        output_dir=dataset_dir,
    )

    result = train_conditional_diffusion_solver(
        TrainingConfig(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            split="train",
            batch_size=2,
            sampling_steps=3,
            max_train_steps=1,
            checkpoint_every_steps=1,
            learning_rate=0.05,
            seed=123,
            device="cpu",
            precision="float32",
        )
    )

    assert result.trained_examples == 2
    assert result.losses and result.losses[-1] > 0
    assert [checkpoint.name for checkpoint in result.checkpoints] == [
        "checkpoint-step-000001.json"
    ]

    checkpoint = json.loads(result.checkpoints[0].read_text())
    assert checkpoint["model_type"] == "custom_conditional_diffusion_solver"
    assert checkpoint["training_step"] == 1
    assert checkpoint["config"]["sampling_steps"] == 3
    assert checkpoint["config"]["device"] == "cpu"


def test_training_config_loads_runtime_and_hardware_settings(tmp_path):
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(tmp_path / "dataset"),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "split": "validation",
                "batch_size": 4,
                "sampling_steps": 5,
                "max_train_steps": 6,
                "checkpoint_every_steps": 2,
                "learning_rate": 0.025,
                "seed": 99,
                "device": "cpu",
                "precision": "float32",
                "num_workers": 0,
            }
        )
    )

    config = load_training_config(config_path)

    assert config.dataset_dir == tmp_path / "dataset"
    assert config.checkpoint_dir == tmp_path / "checkpoints"
    assert config.split == "validation"
    assert config.batch_size == 4
    assert config.sampling_steps == 5
    assert config.max_train_steps == 6
    assert config.checkpoint_every_steps == 2
    assert config.learning_rate == 0.025
    assert config.seed == 99
    assert config.device == "cpu"
    assert config.precision == "float32"
    assert config.num_workers == 0


def test_training_config_controls_checkpoint_cadence(tmp_path):
    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    write_dataset_artifacts(
        config=DatasetConfig(
            width=4,
            height=4,
            split_sizes={
                DatasetSplitName.TRAIN: 1,
                DatasetSplitName.VALIDATION: 0,
                DatasetSplitName.TEST: 0,
            },
            seed_ranges={
                DatasetSplitName.TRAIN: range(20, 30),
                DatasetSplitName.VALIDATION: range(120, 121),
                DatasetSplitName.TEST: range(220, 221),
            },
        ),
        output_dir=dataset_dir,
    )

    result = train_conditional_diffusion_solver(
        TrainingConfig(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            batch_size=1,
            sampling_steps=2,
            max_train_steps=3,
            checkpoint_every_steps=2,
        )
    )

    assert len(result.losses) == 3
    assert [checkpoint.name for checkpoint in result.checkpoints] == [
        "checkpoint-step-000002.json"
    ]
    assert [checkpoint.name for checkpoint in checkpoint_dir.glob("*.json")] == [
        "checkpoint-step-000002.json"
    ]


def test_training_cli_runs_smoke_training_from_config_file(tmp_path, capsys):
    dataset_dir = tmp_path / "dataset"
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "training.json"
    write_dataset_artifacts(
        config=DatasetConfig(
            width=4,
            height=4,
            split_sizes={
                DatasetSplitName.TRAIN: 1,
                DatasetSplitName.VALIDATION: 0,
                DatasetSplitName.TEST: 0,
            },
            seed_ranges={
                DatasetSplitName.TRAIN: range(30, 40),
                DatasetSplitName.VALIDATION: range(130, 131),
                DatasetSplitName.TEST: range(230, 231),
            },
        ),
        output_dir=dataset_dir,
    )
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "split": "train",
                "batch_size": 1,
                "sampling_steps": 2,
                "max_train_steps": 1,
                "checkpoint_every_steps": 1,
                "learning_rate": 0.01,
                "device": "cpu",
                "precision": "float32",
            }
        )
    )

    exit_code = run_training_cli(["--config", str(config_path)])

    assert exit_code == 0
    assert (checkpoint_dir / "checkpoint-step-000001.json").exists()
    report = capsys.readouterr().out
    assert "Conditional Diffusion Solver training complete" in report
    assert "Training steps: 1" in report
    assert "Checkpoints written: 1" in report

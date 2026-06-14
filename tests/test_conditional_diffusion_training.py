import json

import pytest

from dreamaze.dataset import (
    DatasetConfig,
    DatasetSplitName,
    MazeFamily,
    write_dataset_artifacts,
)
from dreamaze.training import DIFFUSERS_MODEL_TYPE
from dreamaze.training import (
    TrainingConfig,
    TrainingExampleArrays,
    _loss_weight_channels,
    _off_path_suppression_loss,
    _path_continuity_loss,
    _weighted_bce_loss,
    _weighted_soft_dice_loss,
    _wall_suppression_loss,
    load_training_config,
    _solution_mask_target_channels,
    _weighted_mse_loss,
    train_conditional_diffusion_solver,
)
from dreamaze.training_cli import run_training_cli


def test_tiny_dataset_artifacts_flow_through_training_and_write_checkpoint(tmp_path):
    _requires_diffusers_runtime()
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
        "checkpoint-step-000001"
    ]

    checkpoint = json.loads((result.checkpoints[0] / "metadata.json").read_text())
    assert checkpoint["model_type"] == DIFFUSERS_MODEL_TYPE
    assert checkpoint["training_step"] == 1
    assert checkpoint["config"]["sampling_steps"] == 3
    assert checkpoint["config"]["device"] == "cpu"
    assert (result.checkpoints[0] / "unet").is_dir()
    assert (result.checkpoints[0] / "scheduler").is_dir()


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
                "positive_loss_weight": 8.0,
                "endpoint_loss_weight": 32.0,
                "mask_bce_loss_weight": 1.5,
                "mask_dice_loss_weight": 0.75,
                "wall_loss_weight": 2.5,
                "path_continuity_loss_weight": 1.25,
                "off_path_loss_weight": 3.5,
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
    assert config.positive_loss_weight == 8.0
    assert config.endpoint_loss_weight == 32.0
    assert config.mask_bce_loss_weight == 1.5
    assert config.mask_dice_loss_weight == 0.75
    assert config.wall_loss_weight == 2.5
    assert config.path_continuity_loss_weight == 1.25
    assert config.off_path_loss_weight == 3.5


def test_training_encodes_solution_mask_targets_symmetrically():
    example = TrainingExampleArrays(
        maze_condition=((1, 1), (1, 1)),
        solution_mask=((0, 1), (1, 0)),
        start_cell=(0, 0),
        goal_cell=(1, 1),
    )

    assert _solution_mask_target_channels(example) == [
        [
            [-1.0, 1.0],
            [1.0, -1.0],
        ]
    ]


def test_training_weights_sparse_path_and_endpoints():
    example = TrainingExampleArrays(
        maze_condition=tuple(tuple(1 for _ in range(5)) for _ in range(5)),
        solution_mask=(
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0),
        ),
        start_cell=(0, 0),
        goal_cell=(1, 1),
    )

    assert _loss_weight_channels(
        example,
        positive_loss_weight=8.0,
        endpoint_loss_weight=32.0,
    ) == [
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 32.0, 8.0, 8.0, 1.0],
            [1.0, 1.0, 1.0, 8.0, 1.0],
            [1.0, 1.0, 1.0, 32.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    ]


def test_weighted_training_loss_is_normalized_by_total_weight():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    loss = _weighted_mse_loss(
        predicted=torch.tensor([[[[2.0, 3.0]]]]),
        target=torch.tensor([[[[0.0, 1.0]]]]),
        weights=torch.tensor([[[[1.0, 3.0]]]]),
    )

    assert loss.item() == pytest.approx(4.0)


def test_weighted_mask_bce_loss_is_normalized_by_total_weight():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    logits = torch.tensor([[[[0.0, 2.0]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])
    weights = torch.tensor([[[[1.0, 3.0]]]])
    loss = _weighted_bce_loss(
        torch=torch,
        logits=logits,
        target=target,
        weights=weights,
    )
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        * weights
    ).sum() / weights.sum()

    assert loss.item() == pytest.approx(expected.item())


def test_weighted_soft_dice_loss_rewards_mask_overlap():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    target = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    weights = torch.ones_like(target)

    good_loss = _weighted_soft_dice_loss(
        torch=torch,
        logits=torch.tensor([[[[-8.0, 8.0, 8.0]]]]),
        target=target,
        weights=weights,
    )
    bad_loss = _weighted_soft_dice_loss(
        torch=torch,
        logits=torch.tensor([[[[8.0, -8.0, -8.0]]]]),
        target=target,
        weights=weights,
    )

    assert good_loss.item() < bad_loss.item()


def test_wall_suppression_loss_penalizes_closed_unlabeled_maze_cells_only():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    clean_logits = torch.tensor([[[[8.0, 8.0, -8.0]]]])
    clean_labels = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    condition = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ]
        ]
    )

    loss = _wall_suppression_loss(
        torch=torch,
        clean_logits=clean_logits,
        condition=condition,
        clean_labels=clean_labels,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-8.0]),
        torch.tensor([0.0]),
    )

    assert loss.item() == pytest.approx(expected.item())


def test_path_continuity_loss_penalizes_broken_label_edges():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    clean_labels = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    complete_path_loss = _path_continuity_loss(
        torch=torch,
        clean_logits=torch.tensor([[[[8.0, 8.0, 8.0]]]]),
        clean_labels=clean_labels,
    )
    broken_path_loss = _path_continuity_loss(
        torch=torch,
        clean_logits=torch.tensor([[[[8.0, -8.0, 8.0]]]]),
        clean_labels=clean_labels,
    )
    expected_broken_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    )

    assert complete_path_loss.item() < broken_path_loss.item()
    assert broken_path_loss.item() == pytest.approx(expected_broken_loss.item())


def test_off_path_suppression_loss_penalizes_non_label_cells_only():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    clean_logits = torch.tensor([[[[8.0, 8.0, -8.0]]]])
    clean_labels = torch.tensor([[[[0.0, 1.0, 0.0]]]])

    loss = _off_path_suppression_loss(
        torch=torch,
        clean_logits=clean_logits,
        clean_labels=clean_labels,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([8.0, -8.0]),
        torch.tensor([0.0, 0.0]),
    )

    assert loss.item() == pytest.approx(expected.item())


def test_training_config_controls_checkpoint_cadence(tmp_path):
    _requires_diffusers_runtime()
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
        "checkpoint-step-000002"
    ]
    assert [checkpoint.name for checkpoint in checkpoint_dir.iterdir()] == [
        "checkpoint-step-000002"
    ]


def test_training_cli_runs_smoke_training_from_config_file(tmp_path, capsys):
    _requires_diffusers_runtime()
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
    assert (checkpoint_dir / "checkpoint-step-000001" / "metadata.json").exists()
    report = capsys.readouterr().out
    assert "Conditional Diffusion Solver training complete" in report
    assert "Training steps: 1" in report
    assert "Checkpoints written: 1" in report


def _requires_diffusers_runtime() -> None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Diffusers runtime is unavailable locally: {error}")

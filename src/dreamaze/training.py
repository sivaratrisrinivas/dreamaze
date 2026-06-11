import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping

from dreamaze.dataset import (
    DatasetSplitName,
    load_dataset_artifact_manifest,
    load_dataset_artifact_shard,
)


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str | Path
    checkpoint_dir: str | Path
    split: str = "train"
    batch_size: int = 8
    sampling_steps: int = 8
    max_train_steps: int = 1
    checkpoint_every_steps: int = 1
    learning_rate: float = 0.01
    seed: int = 0
    device: str = "cpu"
    precision: str = "float32"
    num_workers: int = 0


@dataclass(frozen=True)
class TrainingResult:
    losses: tuple[float, ...]
    checkpoints: tuple[Path, ...]
    trained_examples: int


@dataclass(frozen=True)
class _TrainingExampleArrays:
    maze_condition: tuple[tuple[int, ...], ...]
    solution_mask: tuple[tuple[int, ...], ...]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]


@dataclass
class _ConditionalDiffusionSolver:
    weights: dict[str, float]

    @classmethod
    def initialized(cls, *, rng: Random) -> "_ConditionalDiffusionSolver":
        return cls(
            weights={
                "bias": rng.uniform(-0.05, 0.05),
                "maze_open": rng.uniform(-0.05, 0.05),
                "start": rng.uniform(-0.05, 0.05),
                "goal": rng.uniform(-0.05, 0.05),
                "noisy_mask": rng.uniform(-0.05, 0.05),
                "timestep": rng.uniform(-0.05, 0.05),
            }
        )


def train_conditional_diffusion_solver(config: TrainingConfig) -> TrainingResult:
    _validate_training_config(config)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_training_examples(config)
    rng = Random(config.seed)
    model = _ConditionalDiffusionSolver.initialized(rng=rng)

    losses: list[float] = []
    checkpoints: list[Path] = []
    trained_examples = 0

    for training_step in range(1, config.max_train_steps + 1):
        batch = _training_batch(
            examples=examples, batch_size=config.batch_size, training_step=training_step
        )
        loss = _train_one_step(
            model=model,
            batch=batch,
            training_step=training_step,
            sampling_steps=config.sampling_steps,
            learning_rate=config.learning_rate,
            rng=rng,
        )
        losses.append(loss)
        trained_examples += len(batch)

        if training_step % config.checkpoint_every_steps == 0:
            checkpoints.append(
                _write_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    config=config,
                    model=model,
                    training_step=training_step,
                    loss=loss,
                )
            )

    return TrainingResult(
        losses=tuple(losses),
        checkpoints=tuple(checkpoints),
        trained_examples=trained_examples,
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = json.loads(Path(path).read_text())
    config = TrainingConfig(
        dataset_dir=Path(payload["dataset_dir"]),
        checkpoint_dir=Path(payload["checkpoint_dir"]),
        split=payload.get("split", "train"),
        batch_size=payload["batch_size"],
        sampling_steps=payload["sampling_steps"],
        max_train_steps=payload["max_train_steps"],
        checkpoint_every_steps=payload["checkpoint_every_steps"],
        learning_rate=payload["learning_rate"],
        seed=payload.get("seed", 0),
        device=payload.get("device", "cpu"),
        precision=payload.get("precision", "float32"),
        num_workers=payload.get("num_workers", 0),
    )
    _validate_training_config(config)
    return config


def _load_training_examples(config: TrainingConfig) -> tuple[_TrainingExampleArrays, ...]:
    dataset_dir = Path(config.dataset_dir)
    manifest = load_dataset_artifact_manifest(dataset_dir / "manifest.json")
    requested_split = DatasetSplitName(config.split)
    examples: list[_TrainingExampleArrays] = []

    for shard in manifest.shards:
        if shard.split != requested_split:
            continue
        payload = load_dataset_artifact_shard(dataset_dir / shard.name)
        examples.extend(_examples_from_shard_payload(payload))

    if not examples:
        raise ValueError(f"No Dataset Artifact examples found for {config.split} split")

    return tuple(examples)


def _examples_from_shard_payload(
    payload: Mapping[str, Any]
) -> tuple[_TrainingExampleArrays, ...]:
    return tuple(
        _TrainingExampleArrays(
            maze_condition=_grid(payload["maze_condition"][index]),
            solution_mask=_grid(payload["solution_mask"][index]),
            start_cell=tuple(payload["start_cell"][index]),
            goal_cell=tuple(payload["goal_cell"][index]),
        )
        for index in range(len(payload["maze_condition"]))
    )


def _training_batch(
    *,
    examples: tuple[_TrainingExampleArrays, ...],
    batch_size: int,
    training_step: int,
) -> tuple[_TrainingExampleArrays, ...]:
    start = ((training_step - 1) * batch_size) % len(examples)
    return tuple(examples[(start + offset) % len(examples)] for offset in range(batch_size))


def _train_one_step(
    *,
    model: _ConditionalDiffusionSolver,
    batch: tuple[_TrainingExampleArrays, ...],
    training_step: int,
    sampling_steps: int,
    learning_rate: float,
    rng: Random,
) -> float:
    gradients = {name: 0.0 for name in model.weights}
    total_loss = 0.0
    sample_count = 0
    timestep = 1 + ((training_step - 1) % sampling_steps)
    timestep_feature = timestep / sampling_steps

    for example in batch:
        start_cell = _rendered_cell(example.start_cell)
        goal_cell = _rendered_cell(example.goal_cell)
        for row_index, row in enumerate(example.maze_condition):
            for column_index, is_open in enumerate(row):
                target = float(example.solution_mask[row_index][column_index])
                noisy_mask = _noisy_mask_value(
                    target=target, timestep_feature=timestep_feature, rng=rng
                )
                features = {
                    "bias": 1.0,
                    "maze_open": float(is_open),
                    "start": float((row_index, column_index) == start_cell),
                    "goal": float((row_index, column_index) == goal_cell),
                    "noisy_mask": noisy_mask,
                    "timestep": timestep_feature,
                }
                prediction = _predict(model.weights, features)
                error = prediction - target
                total_loss += error * error
                sample_count += 1
                sigmoid_gradient = prediction * (1.0 - prediction)
                for name, value in features.items():
                    gradients[name] += 2.0 * error * sigmoid_gradient * value

    for name in model.weights:
        model.weights[name] -= learning_rate * gradients[name] / sample_count

    return total_loss / sample_count


def _write_checkpoint(
    *,
    checkpoint_dir: Path,
    config: TrainingConfig,
    model: _ConditionalDiffusionSolver,
    training_step: int,
    loss: float,
) -> Path:
    checkpoint_path = checkpoint_dir / f"checkpoint-step-{training_step:06d}.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "model_type": "custom_conditional_diffusion_solver",
                "training_step": training_step,
                "loss": loss,
                "weights": model.weights,
                "config": _training_config_payload(config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return checkpoint_path


def _training_config_payload(config: TrainingConfig) -> Mapping[str, Any]:
    return {
        "dataset_dir": str(config.dataset_dir),
        "checkpoint_dir": str(config.checkpoint_dir),
        "split": config.split,
        "batch_size": config.batch_size,
        "sampling_steps": config.sampling_steps,
        "max_train_steps": config.max_train_steps,
        "checkpoint_every_steps": config.checkpoint_every_steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": config.device,
        "precision": config.precision,
        "num_workers": config.num_workers,
    }


def _validate_training_config(config: TrainingConfig) -> None:
    DatasetSplitName(config.split)
    if config.batch_size < 1:
        raise ValueError("Training batch size must be positive")
    if config.sampling_steps < 1:
        raise ValueError("Training sampling steps must be positive")
    if config.max_train_steps < 1:
        raise ValueError("Training max steps must be positive")
    if config.checkpoint_every_steps < 1:
        raise ValueError("Training checkpoint cadence must be positive")
    if config.learning_rate <= 0:
        raise ValueError("Training learning rate must be positive")
    if config.num_workers < 0:
        raise ValueError("Training worker count cannot be negative")


def _predict(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    activation = sum(weights[name] * value for name, value in features.items())
    return 1.0 / (1.0 + math.exp(-activation))


def _noisy_mask_value(*, target: float, timestep_feature: float, rng: Random) -> float:
    if rng.random() < timestep_feature * 0.5:
        return 1.0 - target
    return target


def _rendered_cell(cell: tuple[int, int]) -> tuple[int, int]:
    row, column = cell
    return (row * 2 + 1, column * 2 + 1)


def _grid(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in rows)

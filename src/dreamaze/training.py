import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dreamaze.dataset import (
    DatasetSplitName,
    load_dataset_artifact_manifest,
    load_dataset_artifact_shard,
)


DIFFUSERS_MODEL_TYPE = "dreamaze_diffusers_unet2d_conditional_solver"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str | Path
    checkpoint_dir: str | Path
    split: str = "train"
    batch_size: int = 8
    sampling_steps: int = 32
    max_train_steps: int = 1
    checkpoint_every_steps: int = 1
    learning_rate: float = 1e-4
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
class TrainingExampleArrays:
    maze_condition: tuple[tuple[int, ...], ...]
    solution_mask: tuple[tuple[int, ...], ...]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]


def train_conditional_diffusion_solver(config: TrainingConfig) -> TrainingResult:
    """Train Dreamaze's single Conditional Diffusion Solver path.

    This uses Hugging Face Diffusers UNet2DModel as the trainable denoiser. The
    model predicts noise for a Solution Mask while conditioning on the Rendered
    Maze plus Start Cell and Goal Cell channels.
    """
    torch, _, DDPMScheduler, UNet2DModel = _require_diffusion_dependencies()
    _validate_training_config(config)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_training_examples(config)
    sample_size = _model_sample_size(examples)
    device = _torch_device(torch, config.device)
    dtype = _torch_dtype(torch, config.precision)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    model = _build_unet_model(UNet2DModel, sample_size=sample_size).to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.sampling_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    checkpoints: list[Path] = []
    trained_examples = 0

    model.train()
    for training_step in range(1, config.max_train_steps + 1):
        batch = _training_batch(
            examples=examples, batch_size=config.batch_size, training_step=training_step
        )
        condition, target = _batch_tensors(
            torch=torch, batch=batch, device=device, dtype=dtype, sample_size=sample_size
        )
        noise = torch.randn(
            target.shape,
            generator=generator,
            dtype=target.dtype,
        ).to(device)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (target.shape[0],),
            generator=generator,
            dtype=torch.long,
        ).to(device)
        noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
        model_input = torch.cat([noisy_target, condition], dim=1)

        predicted_noise = model(model_input, timesteps).sample
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        trained_examples += len(batch)

        if training_step % config.checkpoint_every_steps == 0:
            checkpoints.append(
                _write_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    config=config,
                    model=model,
                    noise_scheduler=noise_scheduler,
                    training_step=training_step,
                    loss=losses[-1],
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


def load_checkpoint_metadata(checkpoint_path: str | Path) -> Mapping[str, Any]:
    return json.loads((Path(checkpoint_path) / "metadata.json").read_text())


def _require_diffusion_dependencies():
    try:
        import torch
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as error:
        raise RuntimeError(
            "Dreamaze Diffusers training requires torch and diffusers. "
            "Install project dependencies or run this on Hugging Face GPU Jobs."
        ) from error
    return torch, None, DDPMScheduler, UNet2DModel


def _load_training_examples(config: TrainingConfig) -> tuple[TrainingExampleArrays, ...]:
    dataset_dir = Path(config.dataset_dir)
    manifest = load_dataset_artifact_manifest(dataset_dir / "manifest.json")
    requested_split = DatasetSplitName(config.split)
    examples: list[TrainingExampleArrays] = []

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
) -> tuple[TrainingExampleArrays, ...]:
    return tuple(
        TrainingExampleArrays(
            maze_condition=_grid(payload["maze_condition"][index]),
            solution_mask=_grid(payload["solution_mask"][index]),
            start_cell=tuple(payload["start_cell"][index]),
            goal_cell=tuple(payload["goal_cell"][index]),
        )
        for index in range(len(payload["maze_condition"]))
    )


def _build_unet_model(UNet2DModel, *, sample_size: tuple[int, int]):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=4,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )


def _training_batch(
    *,
    examples: tuple[TrainingExampleArrays, ...],
    batch_size: int,
    training_step: int,
) -> tuple[TrainingExampleArrays, ...]:
    start = ((training_step - 1) * batch_size) % len(examples)
    return tuple(examples[(start + offset) % len(examples)] for offset in range(batch_size))


def _batch_tensors(*, torch, batch, device, dtype, sample_size: tuple[int, int]):
    condition_rows = [_condition_channels(example) for example in batch]
    target_rows = [_solution_mask_target_channels(example) for example in batch]
    tensor_dtype = dtype or torch.float32
    condition = torch.tensor(condition_rows, dtype=tensor_dtype, device=device)
    target = torch.tensor(target_rows, dtype=tensor_dtype, device=device)
    condition = _pad_to_sample_size(torch, condition, sample_size)
    target = _pad_to_sample_size(torch, target, sample_size)
    return condition, target


def _solution_mask_target_channels(
    example: TrainingExampleArrays,
) -> list[list[list[float]]]:
    return [
        [
            [1.0 if value else -1.0 for value in row]
            for row in example.solution_mask
        ]
    ]


def _condition_channels(example: TrainingExampleArrays) -> list[list[list[float]]]:
    rows = len(example.maze_condition)
    columns = len(example.maze_condition[0])
    maze_open = [[float(value) for value in row] for row in example.maze_condition]
    start = [[0.0 for _ in range(columns)] for _ in range(rows)]
    goal = [[0.0 for _ in range(columns)] for _ in range(rows)]
    _mark_endpoint_cross(start, _rendered_cell(example.start_cell))
    _mark_endpoint_cross(goal, _rendered_cell(example.goal_cell))
    return [maze_open, start, goal]


def _mark_endpoint_cross(channel: list[list[float]], cell: tuple[int, int]) -> None:
    rows = len(channel)
    columns = len(channel[0])
    cell_row, cell_column = cell
    for row, column in (
        (cell_row, cell_column),
        (cell_row - 1, cell_column),
        (cell_row + 1, cell_column),
        (cell_row, cell_column - 1),
        (cell_row, cell_column + 1),
    ):
        if 0 <= row < rows and 0 <= column < columns:
            channel[row][column] = 1.0


def _write_checkpoint(
    *,
    checkpoint_dir: Path,
    config: TrainingConfig,
    model,
    noise_scheduler,
    training_step: int,
    loss: float,
) -> Path:
    checkpoint_path = checkpoint_dir / f"checkpoint-step-{training_step:06d}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path / "unet")
    noise_scheduler.save_pretrained(checkpoint_path / "scheduler")
    (checkpoint_path / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": DIFFUSERS_MODEL_TYPE,
                "training_step": training_step,
                "loss": loss,
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


def _model_sample_size(examples: tuple[TrainingExampleArrays, ...]) -> tuple[int, int]:
    rows = len(examples[0].maze_condition)
    columns = len(examples[0].maze_condition[0])
    return _ceil_multiple(rows, 8), _ceil_multiple(columns, 8)


def _ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_to_sample_size(torch, tensor, sample_size: tuple[int, int]):
    rows = tensor.shape[-2]
    columns = tensor.shape[-1]
    target_rows, target_columns = sample_size
    return torch.nn.functional.pad(
        tensor, (0, target_columns - columns, 0, target_rows - rows)
    )


def _torch_device(torch, requested_device: str):
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested_device)


def _torch_dtype(torch, precision: str):
    if precision == "float32":
        return torch.float32
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    raise ValueError("Training precision must be float32, float16, or bfloat16")


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
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("Training device must be cpu or cuda")
    if config.precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Training precision must be float32, float16, or bfloat16")


def _rendered_cell(cell: tuple[int, int]) -> tuple[int, int]:
    row, column = cell
    return (row * 2 + 1, column * 2 + 1)


def _grid(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in rows)

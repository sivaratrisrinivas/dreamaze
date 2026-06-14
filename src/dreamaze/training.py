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
    positive_loss_weight: float = 1.0
    endpoint_loss_weight: float = 1.0
    mask_bce_loss_weight: float = 0.0
    mask_dice_loss_weight: float = 0.0
    wall_loss_weight: float = 0.0
    path_continuity_loss_weight: float = 0.0
    off_path_loss_weight: float = 0.0


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
        condition, target, loss_weights = _batch_tensors(
            torch=torch,
            batch=batch,
            device=device,
            dtype=dtype,
            sample_size=sample_size,
            positive_loss_weight=config.positive_loss_weight,
            endpoint_loss_weight=config.endpoint_loss_weight,
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
        noise_loss = _weighted_mse_loss(
            predicted=predicted_noise,
            target=noise,
            weights=loss_weights,
        )
        loss = noise_loss + _clean_mask_auxiliary_loss(
            torch=torch,
            noisy_target=noisy_target,
            predicted_noise=predicted_noise,
            clean_target=target,
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            weights=loss_weights,
            condition=condition,
            bce_loss_weight=config.mask_bce_loss_weight,
            dice_loss_weight=config.mask_dice_loss_weight,
            wall_loss_weight=config.wall_loss_weight,
            path_continuity_loss_weight=config.path_continuity_loss_weight,
            off_path_loss_weight=config.off_path_loss_weight,
        )

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
        positive_loss_weight=payload.get("positive_loss_weight", 1.0),
        endpoint_loss_weight=payload.get("endpoint_loss_weight", 1.0),
        mask_bce_loss_weight=payload.get("mask_bce_loss_weight", 0.0),
        mask_dice_loss_weight=payload.get("mask_dice_loss_weight", 0.0),
        wall_loss_weight=payload.get("wall_loss_weight", 0.0),
        path_continuity_loss_weight=payload.get("path_continuity_loss_weight", 0.0),
        off_path_loss_weight=payload.get("off_path_loss_weight", 0.0),
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


def _batch_tensors(
    *,
    torch,
    batch,
    device,
    dtype,
    sample_size: tuple[int, int],
    positive_loss_weight: float = 1.0,
    endpoint_loss_weight: float = 1.0,
):
    condition_rows = [_condition_channels(example) for example in batch]
    target_rows = [_solution_mask_target_channels(example) for example in batch]
    weight_rows = [
        _loss_weight_channels(
            example,
            positive_loss_weight=positive_loss_weight,
            endpoint_loss_weight=endpoint_loss_weight,
        )
        for example in batch
    ]
    tensor_dtype = dtype or torch.float32
    condition = torch.tensor(condition_rows, dtype=tensor_dtype, device=device)
    target = torch.tensor(target_rows, dtype=tensor_dtype, device=device)
    weights = torch.tensor(weight_rows, dtype=tensor_dtype, device=device)
    condition = _pad_to_sample_size(torch, condition, sample_size)
    target = _pad_to_sample_size(torch, target, sample_size)
    weights = _pad_to_sample_size(torch, weights, sample_size)
    return condition, target, weights


def _solution_mask_target_channels(
    example: TrainingExampleArrays,
) -> list[list[list[float]]]:
    return [
        [
            [1.0 if value else -1.0 for value in row]
            for row in example.solution_mask
        ]
    ]


def _loss_weight_channels(
    example: TrainingExampleArrays,
    *,
    positive_loss_weight: float = 1.0,
    endpoint_loss_weight: float = 1.0,
) -> list[list[list[float]]]:
    rows = len(example.solution_mask)
    columns = len(example.solution_mask[0])
    weights = [[1.0 for _ in range(columns)] for _ in range(rows)]
    for row_index, row in enumerate(example.solution_mask):
        for column_index, value in enumerate(row):
            if value:
                weights[row_index][column_index] = positive_loss_weight
    start_row, start_column = _rendered_cell(example.start_cell)
    goal_row, goal_column = _rendered_cell(example.goal_cell)
    weights[start_row][start_column] = max(
        weights[start_row][start_column], endpoint_loss_weight
    )
    weights[goal_row][goal_column] = max(
        weights[goal_row][goal_column], endpoint_loss_weight
    )
    return [weights]


def _weighted_mse_loss(*, predicted, target, weights):
    return ((predicted - target) ** 2 * weights).sum() / weights.sum()


def _clean_mask_auxiliary_loss(
    *,
    torch,
    noisy_target,
    predicted_noise,
    clean_target,
    timesteps,
    noise_scheduler,
    weights,
    condition,
    bce_loss_weight: float,
    dice_loss_weight: float,
    wall_loss_weight: float,
    path_continuity_loss_weight: float,
    off_path_loss_weight: float,
):
    if (
        bce_loss_weight == 0
        and dice_loss_weight == 0
        and wall_loss_weight == 0
        and path_continuity_loss_weight == 0
        and off_path_loss_weight == 0
    ):
        return predicted_noise.new_tensor(0.0)

    clean_logits = _predicted_clean_target_from_noise(
        noisy_target=noisy_target,
        predicted_noise=predicted_noise,
        timesteps=timesteps,
        noise_scheduler=noise_scheduler,
    )
    clean_labels = (clean_target + 1.0) / 2.0
    auxiliary_loss = predicted_noise.new_tensor(0.0)
    if bce_loss_weight:
        auxiliary_loss = auxiliary_loss + bce_loss_weight * _weighted_bce_loss(
            torch=torch,
            logits=clean_logits,
            target=clean_labels,
            weights=weights,
        )
    if dice_loss_weight:
        auxiliary_loss = auxiliary_loss + dice_loss_weight * _weighted_soft_dice_loss(
            torch=torch,
            logits=clean_logits,
            target=clean_labels,
            weights=weights,
        )
    if wall_loss_weight:
        auxiliary_loss = auxiliary_loss + wall_loss_weight * _wall_suppression_loss(
            torch=torch,
            clean_logits=clean_logits,
            condition=condition,
            clean_labels=clean_labels,
        )
    if path_continuity_loss_weight:
        auxiliary_loss = (
            auxiliary_loss
            + path_continuity_loss_weight
            * _path_continuity_loss(
                torch=torch,
                clean_logits=clean_logits,
                clean_labels=clean_labels,
            )
        )
    if off_path_loss_weight:
        auxiliary_loss = (
            auxiliary_loss
            + off_path_loss_weight
            * _off_path_suppression_loss(
                torch=torch,
                clean_logits=clean_logits,
                clean_labels=clean_labels,
            )
        )
    return auxiliary_loss


def _predicted_clean_target_from_noise(
    *, noisy_target, predicted_noise, timesteps, noise_scheduler
):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_target.device,
        dtype=noisy_target.dtype,
    )
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    return (
        noisy_target - beta_prod_t.sqrt() * predicted_noise
    ) / alpha_prod_t.sqrt().clamp_min(1e-12)


def _weighted_bce_loss(*, torch, logits, target, weights):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    return (loss * weights).sum() / weights.sum()


def _weighted_soft_dice_loss(*, torch, logits, target, weights, smooth: float = 1.0):
    probability = torch.sigmoid(logits)
    intersection = (probability * target * weights).sum()
    total = ((probability + target) * weights).sum()
    return 1.0 - ((2.0 * intersection + smooth) / (total + smooth))


def _wall_suppression_loss(*, torch, clean_logits, condition, clean_labels):
    maze_open = condition[:, :1]
    wall_weights = ((maze_open <= 0.0) & (clean_labels <= 0.0)).to(
        dtype=clean_logits.dtype
    )
    wall_target = torch.zeros_like(clean_logits)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        clean_logits,
        wall_target,
        reduction="none",
    )
    return (loss * wall_weights).sum() / wall_weights.sum().clamp_min(1.0)


def _path_continuity_loss(*, torch, clean_logits, clean_labels):
    label_cells = clean_labels > 0.0
    horizontal_edges = label_cells[:, :, :, :-1] & label_cells[:, :, :, 1:]
    vertical_edges = label_cells[:, :, :-1, :] & label_cells[:, :, 1:, :]

    horizontal_logits = clean_logits[:, :, :, :-1] + clean_logits[:, :, :, 1:]
    vertical_logits = clean_logits[:, :, :-1, :] + clean_logits[:, :, 1:, :]
    horizontal_weights = horizontal_edges.to(dtype=clean_logits.dtype)
    vertical_weights = vertical_edges.to(dtype=clean_logits.dtype)
    horizontal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        horizontal_logits,
        torch.ones_like(horizontal_logits),
        reduction="none",
    )
    vertical_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        vertical_logits,
        torch.ones_like(vertical_logits),
        reduction="none",
    )
    edge_count = horizontal_weights.sum() + vertical_weights.sum()
    return (
        (horizontal_loss * horizontal_weights).sum()
        + (vertical_loss * vertical_weights).sum()
    ) / edge_count.clamp_min(1.0)


def _off_path_suppression_loss(*, torch, clean_logits, clean_labels):
    off_path_weights = (clean_labels <= 0.0).to(dtype=clean_logits.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        clean_logits,
        torch.zeros_like(clean_logits),
        reduction="none",
    )
    return (loss * off_path_weights).sum() / off_path_weights.sum().clamp_min(1.0)


def _condition_channels(example: TrainingExampleArrays) -> list[list[list[float]]]:
    rows = len(example.maze_condition)
    columns = len(example.maze_condition[0])
    maze_open = [[float(value) for value in row] for row in example.maze_condition]
    start = [[0.0 for _ in range(columns)] for _ in range(rows)]
    goal = [[0.0 for _ in range(columns)] for _ in range(rows)]
    start_row, start_column = _rendered_cell(example.start_cell)
    goal_row, goal_column = _rendered_cell(example.goal_cell)
    start[start_row][start_column] = 1.0
    goal[goal_row][goal_column] = 1.0
    return [maze_open, start, goal]


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
        "positive_loss_weight": config.positive_loss_weight,
        "endpoint_loss_weight": config.endpoint_loss_weight,
        "mask_bce_loss_weight": config.mask_bce_loss_weight,
        "mask_dice_loss_weight": config.mask_dice_loss_weight,
        "wall_loss_weight": config.wall_loss_weight,
        "path_continuity_loss_weight": config.path_continuity_loss_weight,
        "off_path_loss_weight": config.off_path_loss_weight,
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
    if config.positive_loss_weight <= 0:
        raise ValueError("Training positive loss weight must be positive")
    if config.endpoint_loss_weight <= 0:
        raise ValueError("Training endpoint loss weight must be positive")
    if config.mask_bce_loss_weight < 0:
        raise ValueError("Training mask BCE loss weight cannot be negative")
    if config.mask_dice_loss_weight < 0:
        raise ValueError("Training mask Dice loss weight cannot be negative")
    if config.wall_loss_weight < 0:
        raise ValueError("Training wall loss weight cannot be negative")
    if config.path_continuity_loss_weight < 0:
        raise ValueError("Training path continuity loss weight cannot be negative")
    if config.off_path_loss_weight < 0:
        raise ValueError("Training off-path loss weight cannot be negative")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("Training device must be cpu or cuda")
    if config.precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Training precision must be float32, float16, or bfloat16")


def _rendered_cell(cell: tuple[int, int]) -> tuple[int, int]:
    row, column = cell
    return (row * 2 + 1, column * 2 + 1)


def _grid(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in rows)

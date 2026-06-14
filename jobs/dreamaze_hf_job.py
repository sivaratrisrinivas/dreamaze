# /// script
# dependencies = [
#   "diffusers>=0.36",
#   "huggingface-hub>=0.36",
#   "safetensors>=0.4",
#   "torch>=2.3",
# ]
# ///

import argparse
import json
import os
from pathlib import Path

from dreamaze.dataset_cli import (
    first_dataset_size_config,
    larger_dataset_config,
    tiny_dataset_config,
)
from dreamaze.dataset import MazeFamily, write_dataset_artifacts
from dreamaze.evaluation import EvaluationConfig, evaluate_conditional_diffusion_solver
from dreamaze.training import TrainingConfig, train_conditional_diffusion_solver


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    evaluation_output = Path(args.evaluation_output)
    run_dir = _common_parent(dataset_dir, checkpoint_dir, evaluation_output)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_preset is not None:
        dataset_config = _dataset_config_for_preset(args.dataset_preset)
        if args.maze_family is not None:
            dataset_config = _single_family_dataset_config(
                dataset_config=dataset_config,
                maze_family=MazeFamily(args.maze_family),
            )
        manifest = write_dataset_artifacts(
            config=dataset_config,
            output_dir=dataset_dir,
        )
        print(
            "Dataset Artifact ready: "
            f"train={manifest.split_counts['train']} "
            f"validation={manifest.split_counts['validation']} "
            f"test={manifest.split_counts['test']}"
        )
        print(
            "Dataset Maze Families: "
            + ", ".join(maze_family.value for maze_family in dataset_config.maze_families)
        )

    training_config = TrainingConfig(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        split="train",
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        max_train_steps=args.max_train_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        learning_rate=args.learning_rate,
        seed=args.training_seed,
        device=args.device,
        precision=args.precision,
        num_workers=args.num_workers,
        positive_loss_weight=args.positive_loss_weight,
        endpoint_loss_weight=args.endpoint_loss_weight,
        mask_bce_loss_weight=args.mask_bce_loss_weight,
        mask_dice_loss_weight=args.mask_dice_loss_weight,
        wall_loss_weight=args.wall_loss_weight,
        path_continuity_loss_weight=args.path_continuity_loss_weight,
        off_path_loss_weight=args.off_path_loss_weight,
    )
    training_result = train_conditional_diffusion_solver(training_config)
    latest_checkpoint = training_result.checkpoints[-1]
    print(f"Latest checkpoint: {latest_checkpoint}")

    evaluation_config = EvaluationConfig(
        dataset_dir=dataset_dir,
        checkpoint_path=latest_checkpoint,
        split=args.eval_split,
        sampling_steps=args.sampling_steps,
        retry_count=args.retry_count,
        seed=args.evaluation_seed,
        report_path=evaluation_output,
        device=args.device,
        precision=args.precision,
    )
    evaluation_result = evaluate_conditional_diffusion_solver(evaluation_config)
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(evaluation_result.to_json())
    print(f"Evaluation report: {evaluation_output}")
    print(
        "Valid-Solution Rate: "
        f"{evaluation_result.single_sample_success.valid_solution_rate:.6f}"
    )
    print(
        "Single-Sample Success: "
        f"{evaluation_result.single_sample_success.valid_count}/"
        f"{evaluation_result.single_sample_success.evaluated_examples}"
    )
    print("Target Valid-Solution Rate: 0.800000-0.900000")
    print(f"Failure reasons: {dict(evaluation_result.failure_reason_counts)}")
    print(
        "Endpoint inclusion: "
        f"start={evaluation_result.endpoint_inclusion['start_cell_inclusion_rate']:.6f} "
        f"goal={evaluation_result.endpoint_inclusion['goal_cell_inclusion_rate']:.6f} "
        f"both={evaluation_result.endpoint_inclusion['both_endpoints_inclusion_rate']:.6f}"
    )
    structure = evaluation_result.structure_stats
    print(
        "Structure diagnostics (primary): "
        f"components_mean={structure.get('connected_component_mean', 0):.3f} "
        f"wall_crossings_mean={structure.get('wall_crossing_count_mean', 0):.3f} "
        f"branch_violations_mean={structure.get('extra_branch_violation_mean', 0):.3f} "
        f"same_comp_rate={structure.get('endpoints_in_same_component_rate', 0):.3f}"
    )
    best_threshold = max(
        evaluation_result.threshold_calibration,
        key=lambda item: (
            item["valid_solution_rate"],
            item["both_endpoints_inclusion_rate"],
            item["mask_overlap"],
        ),
    )
    print(
        "Best threshold calibration: "
        f"threshold={best_threshold['threshold']:.2f} "
        f"valid_solution_rate={best_threshold['valid_solution_rate']:.6f} "
        f"both_endpoints={best_threshold['both_endpoints_inclusion_rate']:.6f}"
    )
    print(
        "  best-thresh structure: "
        f"components_mean={best_threshold.get('connected_component_mean', 0):.3f} "
        f"wall_crossings_mean={best_threshold.get('wall_crossing_count_mean', 0):.3f} "
        f"branch_viol_mean={best_threshold.get('extra_branch_violation_mean', 0):.3f} "
        f"same_comp_rate={best_threshold.get('endpoints_in_same_component_rate', 0):.3f}"
    )

    _write_run_summary(
        run_dir=run_dir,
        args=args,
        latest_checkpoint=latest_checkpoint,
        evaluation_output=evaluation_output,
    )
    if args.output_repo is not None:
        _upload_run_artifacts(
            run_dir=run_dir,
            repo_id=args.output_repo,
            repo_type=args.output_repo_type,
            path_in_repo=args.output_path_in_repo,
        )
    if args.model_output_repo is not None:
        _upload_checkpoint(
            checkpoint_path=latest_checkpoint,
            repo_id=args.model_output_repo,
            path_in_repo=(
                args.model_output_path_in_repo.rstrip("/")
                + "/"
                + latest_checkpoint.name
            ),
        )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dreamaze_hf_job.py")
    parser.add_argument("--dataset-preset", choices=("tiny", "first", "larger"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--evaluation-output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sampling-steps", type=int, required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--positive-loss-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-bce-loss-weight", type=float, default=0.0)
    parser.add_argument("--mask-dice-loss-weight", type=float, default=0.0)
    parser.add_argument("--wall-loss-weight", type=float, default=0.0)
    parser.add_argument("--path-continuity-loss-weight", type=float, default=0.0)
    parser.add_argument("--off-path-loss-weight", type=float, default=0.0)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--retry-count", type=int, required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--maze-family", choices=("kruskal", "wilson"))
    parser.add_argument("--output-repo")
    parser.add_argument("--output-repo-type", default="dataset")
    parser.add_argument("--output-path-in-repo", default="dreamaze-runs/latest")
    parser.add_argument("--model-output-repo")
    parser.add_argument(
        "--model-output-path-in-repo",
        default="checkpoints/latest",
    )
    return parser


def _dataset_config_for_preset(preset: str):
    if preset == "tiny":
        return tiny_dataset_config()
    if preset == "first":
        return first_dataset_size_config()
    if preset == "larger":
        return larger_dataset_config()
    raise ValueError(f"Unknown Dataset Builder preset: {preset}")


def _single_family_dataset_config(*, dataset_config, maze_family: MazeFamily):
    return type(dataset_config)(
        width=dataset_config.width,
        height=dataset_config.height,
        maze_families=(maze_family,),
        split_sizes=dataset_config.split_sizes,
        seed_ranges=dataset_config.seed_ranges,
        border_endpoint_pair_rule=dataset_config.border_endpoint_pair_rule,
        minimum_path_length=dataset_config.minimum_path_length,
        output_format=dataset_config.output_format,
        shard_size=dataset_config.shard_size,
        write_preview_images=dataset_config.write_preview_images,
        max_rejections_per_split=dataset_config.max_rejections_per_split,
    )


def _write_run_summary(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    latest_checkpoint: Path,
    evaluation_output: Path,
) -> None:
    summary = {
        "dataset_preset": args.dataset_preset,
        "dataset_dir": args.dataset_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "latest_checkpoint": str(latest_checkpoint),
        "evaluation_output": str(evaluation_output),
        "batch_size": args.batch_size,
        "sampling_steps": args.sampling_steps,
        "max_train_steps": args.max_train_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "positive_loss_weight": args.positive_loss_weight,
        "endpoint_loss_weight": args.endpoint_loss_weight,
        "mask_bce_loss_weight": args.mask_bce_loss_weight,
        "mask_dice_loss_weight": args.mask_dice_loss_weight,
        "wall_loss_weight": args.wall_loss_weight,
        "path_continuity_loss_weight": args.path_continuity_loss_weight,
        "off_path_loss_weight": args.off_path_loss_weight,
        "eval_split": args.eval_split,
        "device": args.device,
        "precision": args.precision,
        "maze_family": args.maze_family,
    }
    (run_dir / "hf-job-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _upload_run_artifacts(
    *, run_dir: Path, repo_id: str, repo_type: str, path_in_repo: str
) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required when --output-repo is set")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(run_dir),
        path_in_repo=path_in_repo,
    )
    print(f"Uploaded run artifacts to {repo_id}/{path_in_repo}")


def _upload_checkpoint(*, checkpoint_path: Path, repo_id: str, path_in_repo: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required when --model-output-repo is set")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint_path),
        path_in_repo=path_in_repo,
    )
    print(f"Uploaded trained checkpoint to {repo_id}/{path_in_repo}")


def _common_parent(*paths: Path) -> Path:
    resolved = [path.resolve() for path in paths]
    common = os.path.commonpath(
        [str(path.parent if path.suffix else path) for path in resolved]
    )
    return Path(common)


if __name__ == "__main__":
    raise SystemExit(main())

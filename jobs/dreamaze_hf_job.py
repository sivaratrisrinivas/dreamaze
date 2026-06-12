# /// script
# dependencies = [
#   "dreamaze @ git+https://github.com/sivaratrisrinivas/dreamaze.git",
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

from dreamaze.dataset_cli import first_dataset_size_config, tiny_dataset_config
from dreamaze.dataset import write_dataset_artifacts
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
        dataset_config = (
            tiny_dataset_config()
            if args.dataset_preset == "tiny"
            else first_dataset_size_config()
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

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dreamaze_hf_job.py")
    parser.add_argument("--dataset-preset", choices=("tiny", "first"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--evaluation-output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sampling-steps", type=int, required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--retry-count", type=int, required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--output-repo")
    parser.add_argument("--output-repo-type", default="dataset")
    parser.add_argument("--output-path-in-repo", default="dreamaze-runs/latest")
    return parser


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
        "eval_split": args.eval_split,
        "device": args.device,
        "precision": args.precision,
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


def _common_parent(*paths: Path) -> Path:
    resolved = [path.resolve() for path in paths]
    common = os.path.commonpath(
        [str(path.parent if path.suffix else path) for path in resolved]
    )
    return Path(common)


if __name__ == "__main__":
    raise SystemExit(main())

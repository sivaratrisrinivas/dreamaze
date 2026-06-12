import argparse
from collections.abc import Sequence
from pathlib import Path

from dreamaze.evaluation import (
    evaluate_conditional_diffusion_solver,
    load_evaluation_config,
)


def run_evaluation_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dreamaze-evaluate")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_evaluation_config(args.config)
    result = evaluate_conditional_diffusion_solver(config)
    if config.report_path is not None:
        Path(config.report_path).write_text(result.to_json())

    print("Conditional Diffusion Solver evaluation complete")
    print("Official score: Single-Sample Success")
    print(f"Dataset split: {result.dataset_split}")
    print(f"Evaluated examples: {result.evaluated_examples}")
    print(
        "Valid-Solution Rate: "
        f"{result.single_sample_success.valid_solution_rate:.6f}"
    )
    if result.retry_success is not None:
        print(
            "Retry Success: "
            f"{result.retry_success.valid_solution_rate:.6f} "
            "(excluded from official score)"
        )
    if config.report_path is not None:
        print(f"Report written: {config.report_path}")
    return 0


def main() -> int:
    return run_evaluation_cli()


if __name__ == "__main__":
    raise SystemExit(main())

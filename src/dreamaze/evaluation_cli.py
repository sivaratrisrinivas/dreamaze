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
    print("Target Valid-Solution Rate: 0.800000-0.900000")
    print(
        "Single-Sample Success: "
        f"{result.single_sample_success.valid_count}/"
        f"{result.single_sample_success.evaluated_examples}"
    )
    print(
        "Start Cell inclusion: "
        f"{result.endpoint_inclusion['start_cell_inclusion_rate']:.6f}"
    )
    print(
        "Goal Cell inclusion: "
        f"{result.endpoint_inclusion['goal_cell_inclusion_rate']:.6f}"
    )
    print(
        "Both endpoints inclusion: "
        f"{result.endpoint_inclusion['both_endpoints_inclusion_rate']:.6f}"
    )
    print(
        "Mask overlap excluding endpoints: "
        f"{result.endpoint_inclusion['mask_overlap_excluding_endpoints']:.6f}"
    )
    print(
        "Start Cell raw value: "
        f"{result.endpoint_raw_values['start_cell_raw_mean']:.6f}"
    )
    print(
        "Goal Cell raw value: "
        f"{result.endpoint_raw_values['goal_cell_raw_mean']:.6f}"
    )
    if result.retry_success is not None:
        print(
            "Retry Success: "
            f"{result.retry_success.valid_solution_rate:.6f} "
            "(excluded from official score)"
        )
    print(f"Failure reasons: {dict(result.failure_reason_counts)}")
    if config.report_path is not None:
        print(f"Report written: {config.report_path}")
    return 0


def main() -> int:
    return run_evaluation_cli()


if __name__ == "__main__":
    raise SystemExit(main())

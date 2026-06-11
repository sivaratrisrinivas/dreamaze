import argparse
from collections.abc import Sequence

from dreamaze.training import load_training_config, train_conditional_diffusion_solver


def run_training_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dreamaze-train")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    result = train_conditional_diffusion_solver(config)

    print("Conditional Diffusion Solver training complete")
    print(f"Training steps: {len(result.losses)}")
    print(f"Trained examples: {result.trained_examples}")
    print(f"Final loss: {result.losses[-1]:.6f}")
    print(f"Checkpoints written: {len(result.checkpoints)}")
    if result.checkpoints:
        print(f"Latest checkpoint: {result.checkpoints[-1]}")
    return 0


def main() -> int:
    return run_training_cli()


if __name__ == "__main__":
    raise SystemExit(main())

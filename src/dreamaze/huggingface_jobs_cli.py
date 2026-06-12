import argparse
import shlex
import subprocess
from collections.abc import Sequence

from dreamaze.huggingface_jobs import (
    build_huggingface_job_command,
    load_huggingface_job_config,
)


def run_huggingface_jobs_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dreamaze-hf-job")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Hugging Face Jobs command without launching it.",
    )
    args = parser.parse_args(argv)

    config = load_huggingface_job_config(args.config)
    command = build_huggingface_job_command(config)

    if args.dry_run:
        print("Hugging Face Jobs dry run")
        print(shlex.join(command))
        return 0

    return subprocess.run(command, check=False).returncode


def main() -> int:
    return run_huggingface_jobs_cli()


if __name__ == "__main__":
    raise SystemExit(main())

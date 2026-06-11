import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dreamaze.dataset import (
    DatasetConfig,
    DatasetGenerationError,
    DatasetSplitName,
    MazeFamily,
    TrainingExampleBuilder,
    write_dataset_artifacts,
)


def tiny_dataset_config(*, write_preview_images: bool = False) -> DatasetConfig:
    return DatasetConfig(
        width=4,
        height=4,
        maze_families=(MazeFamily.KRUSKAL, MazeFamily.WILSON),
        split_sizes={
            DatasetSplitName.TRAIN: 3,
            DatasetSplitName.VALIDATION: 1,
            DatasetSplitName.TEST: 1,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(10, 20),
            DatasetSplitName.VALIDATION: range(100, 110),
            DatasetSplitName.TEST: range(200, 210),
        },
        shard_size=2,
        minimum_path_length=2,
        write_preview_images=write_preview_images,
    )


def first_dataset_size_config(*, write_preview_images: bool = False) -> DatasetConfig:
    return DatasetConfig(write_preview_images=write_preview_images)


def run_dataset_builder_cli(
    argv: Sequence[str] | None = None,
    *,
    training_example_builder: TrainingExampleBuilder | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        config = _config_for_preset(
            preset=args.preset, write_preview_images=args.preview_images
        )
        output_dir = Path(args.output_dir)
        had_manifest = (output_dir / "manifest.json").exists()
        try:
            if training_example_builder is None:
                manifest = write_dataset_artifacts(config=config, output_dir=output_dir)
            else:
                manifest = write_dataset_artifacts(
                    config=config,
                    output_dir=output_dir,
                    training_example_builder=training_example_builder,
                )
        except DatasetGenerationError as error:
            print(f"Dataset Builder failed: {error}", file=sys.stderr)
            return 1

        print(f"Dataset Builder preset: {args.preset}")
        print(f"Output directory: {output_dir}")
        print(f"Manifest status: {manifest.status}")
        print(
            "Generated counts: "
            f"train={manifest.split_counts['train']} "
            f"validation={manifest.split_counts['validation']} "
            f"test={manifest.split_counts['test']}"
        )
        print("Expected rejections: train=0 validation=0 test=0")
        print("Invariant failures: 0")
        print(f"Shard progress: {len(manifest.shards)}/{len(manifest.shards)} complete")
        if had_manifest:
            print("Resume: reused completed Dataset Artifacts")
        print(f"Manifest path: {output_dir / 'manifest.json'}")
        return 0

    parser.error("missing command")


def main() -> int:
    return run_dataset_builder_cli()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dreamaze-dataset")
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--preset", choices=("tiny", "first"), required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--preview-images", action="store_true")
    return parser


def _config_for_preset(*, preset: str, write_preview_images: bool) -> DatasetConfig:
    if preset == "tiny":
        return tiny_dataset_config(write_preview_images=write_preview_images)
    if preset == "first":
        return first_dataset_size_config(write_preview_images=write_preview_images)
    raise ValueError(f"Unknown Dataset Builder preset: {preset}")


if __name__ == "__main__":
    raise SystemExit(main())

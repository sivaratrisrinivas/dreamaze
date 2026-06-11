import json
from zipfile import ZipFile

from dreamaze.dataset import (
    DatasetConfig,
    DatasetGenerationError,
    DatasetSplitName,
    MazeFamily,
    load_dataset_artifact_manifest,
    load_dataset_artifact_shard,
    write_dataset_artifacts,
)


def test_writes_sharded_dataset_artifacts_with_manifest(tmp_path):
    config = DatasetConfig(
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
    )

    manifest = write_dataset_artifacts(config=config, output_dir=tmp_path)

    assert manifest.status == "complete"
    assert manifest.split_counts == {"train": 3, "validation": 1, "test": 1}
    assert [shard.name for shard in manifest.shards] == [
        "train-00000.npz",
        "train-00001.npz",
        "validation-00000.npz",
        "test-00000.npz",
    ]
    assert (tmp_path / "manifest.json").exists()
    with ZipFile(tmp_path / "train-00000.npz") as artifact:
        assert sorted(artifact.namelist()) == [
            "arrays.json",
            "goal_cell.npy",
            "maze_condition.npy",
            "seed.npy",
            "solution_mask.npy",
            "start_cell.npy",
        ]

    first_train_shard = load_dataset_artifact_shard(tmp_path / "train-00000.npz")

    assert first_train_shard["split"] == ["train", "train"]
    assert first_train_shard["seed"] == [10, 11]
    assert first_train_shard["maze_family"] == ["kruskal", "wilson"]
    assert len(first_train_shard["maze_condition"]) == 2
    assert len(first_train_shard["solution_mask"]) == 2
    assert first_train_shard["start_cell"][0] != first_train_shard["goal_cell"][0]
    assert first_train_shard["metadata"]["width"] == [4, 4]


def test_manifest_records_config_integrity_counts_and_seeds(tmp_path):
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 2,
            DatasetSplitName.VALIDATION: 1,
            DatasetSplitName.TEST: 1,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(20, 30),
            DatasetSplitName.VALIDATION: range(120, 130),
            DatasetSplitName.TEST: range(220, 230),
        },
        shard_size=2,
    )

    written_manifest = write_dataset_artifacts(config=config, output_dir=tmp_path)
    loaded_manifest = load_dataset_artifact_manifest(tmp_path / "manifest.json")

    assert loaded_manifest == written_manifest
    assert loaded_manifest.config["width"] == 4
    assert loaded_manifest.config["seed_ranges"]["validation"] == {
        "start": 120,
        "stop": 130,
        "step": 1,
    }
    assert loaded_manifest.seeds == {
        "train": (20, 21),
        "validation": (120,),
        "test": (220,),
    }

    raw_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert raw_manifest["status"] == "complete"
    assert raw_manifest["split_names"] == ["train", "validation", "test"]
    assert raw_manifest["shards"][0]["sha256"] == loaded_manifest.shards[0].sha256
    assert len(raw_manifest["shards"][0]["sha256"]) == 64


def test_resume_skips_completed_dataset_artifact_shards(tmp_path):
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 2,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(30, 40),
            DatasetSplitName.VALIDATION: range(130, 131),
            DatasetSplitName.TEST: range(230, 231),
        },
        shard_size=2,
    )

    first_manifest = write_dataset_artifacts(config=config, output_dir=tmp_path)

    def fail_if_called(*, seed, config):
        raise AssertionError("resume regenerated a completed shard")

    resumed_manifest = write_dataset_artifacts(
        config=config,
        output_dir=tmp_path,
        training_example_builder=fail_if_called,
    )

    assert resumed_manifest == first_manifest
    assert [path.name for path in sorted(tmp_path.glob("*.npz"))] == [
        "train-00000.npz"
    ]
    assert load_dataset_artifact_shard(tmp_path / "train-00000.npz")["seed"] == [
        30,
        31,
    ]


def test_completed_manifest_with_missing_shard_fails_integrity_check(tmp_path):
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 1,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(35, 40),
            DatasetSplitName.VALIDATION: range(135, 136),
            DatasetSplitName.TEST: range(235, 236),
        },
    )
    manifest = write_dataset_artifacts(config=config, output_dir=tmp_path)
    (tmp_path / manifest.shards[0].name).unlink()

    try:
        write_dataset_artifacts(config=config, output_dir=tmp_path)
    except DatasetGenerationError as error:
        assert "missing Dataset Artifact shard" in str(error)
    else:
        raise AssertionError("missing shard did not stop resume")


def test_incomplete_dataset_artifact_manifest_fails_clearly(tmp_path):
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 1,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(40, 50),
            DatasetSplitName.VALIDATION: range(140, 141),
            DatasetSplitName.TEST: range(240, 241),
        },
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "incomplete", "config": {}})
    )

    try:
        write_dataset_artifacts(config=config, output_dir=tmp_path)
    except DatasetGenerationError as error:
        assert "incomplete Dataset Artifact build" in str(error)
    else:
        raise AssertionError("incomplete manifest did not stop artifact writing")


def test_preview_images_are_separate_from_dataset_artifacts(tmp_path):
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 1,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(50, 60),
            DatasetSplitName.VALIDATION: range(150, 151),
            DatasetSplitName.TEST: range(250, 251),
        },
        write_preview_images=True,
    )

    manifest = write_dataset_artifacts(config=config, output_dir=tmp_path)

    preview_files = sorted((tmp_path / "previews").glob("*.pgm"))
    assert [preview.name for preview in preview_files] == ["train-000050.pgm"]
    assert not list(tmp_path.glob("*.pgm"))
    assert "preview_image" not in load_dataset_artifact_shard(
        tmp_path / manifest.shards[0].name
    )

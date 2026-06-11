from dataclasses import replace

from dreamaze.dataset import (
    DatasetConfig,
    DatasetInvariantError,
    DatasetRejectionLimitError,
    DatasetSplitName,
    MazeFamily,
    build_dataset_splits,
    build_training_example,
)
from dreamaze.validation import validate_solution_mask


def test_builds_tiny_deterministic_dataset_splits_from_fixed_seed_ranges():
    config = DatasetConfig(
        width=4,
        height=4,
        maze_families=(MazeFamily.KRUSKAL, MazeFamily.WILSON),
        split_sizes={
            DatasetSplitName.TRAIN: 3,
            DatasetSplitName.VALIDATION: 2,
            DatasetSplitName.TEST: 2,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(100, 120),
            DatasetSplitName.VALIDATION: range(200, 220),
            DatasetSplitName.TEST: range(300, 320),
        },
        minimum_path_length=2,
    )

    splits = build_dataset_splits(config=config)
    repeated = build_dataset_splits(config=config)

    assert splits == repeated
    assert [example.seed for example in splits.train.examples] == [100, 101, 102]
    assert [example.seed for example in splits.validation.examples] == [200, 201]
    assert [example.seed for example in splits.test.examples] == [300, 301]
    assert [example.maze_family for example in splits.train.examples] == [
        MazeFamily.KRUSKAL,
        MazeFamily.WILSON,
        MazeFamily.KRUSKAL,
    ]

    all_examples = (
        splits.train.examples + splits.validation.examples + splits.test.examples
    )
    assert len({example.seed for example in all_examples}) == len(all_examples)

    for split in (splits.train, splits.validation, splits.test):
        for example in split.examples:
            assert example.split == split.name.value
            assert example.metadata["path_length"] >= config.minimum_path_length
            result = validate_solution_mask(
                grid_maze=example.maze_condition.rendered_maze,
                solution_mask=example.training_label,
                start_cell=example.rendered_start_cell,
                goal_cell=example.rendered_goal_cell,
            )
            assert result.valid is True


def test_dataset_splits_stop_on_duplicate_split_seed():
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 1,
            DatasetSplitName.VALIDATION: 1,
            DatasetSplitName.TEST: 1,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(10, 12),
            DatasetSplitName.VALIDATION: range(11, 13),
            DatasetSplitName.TEST: range(20, 22),
        },
    )

    try:
        build_dataset_splits(config=config)
    except DatasetInvariantError as error:
        assert "Duplicate split seed 11" in str(error)
    else:
        raise AssertionError("duplicate split seed did not stop Dataset Split generation")


def test_minimum_path_length_rejections_are_replaced_until_split_is_full():
    config = DatasetConfig(
        width=4,
        height=4,
        maze_families=(MazeFamily.KRUSKAL,),
        split_sizes={
            DatasetSplitName.TRAIN: 2,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(0, 6),
            DatasetSplitName.VALIDATION: range(100, 101),
            DatasetSplitName.TEST: range(200, 201),
        },
        minimum_path_length=12,
    )

    splits = build_dataset_splits(config=config)

    assert [example.seed for example in splits.train.examples] == [1, 4]
    assert splits.train.rejected_seeds == (0, 2, 3)
    assert all(
        example.metadata["path_length"] >= 12 for example in splits.train.examples
    )


def test_dataset_splits_stop_on_invariant_failure():
    config = DatasetConfig(
        width=4,
        height=4,
        split_sizes={
            DatasetSplitName.TRAIN: 1,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(10, 12),
            DatasetSplitName.VALIDATION: range(20, 21),
            DatasetSplitName.TEST: range(30, 31),
        },
    )

    def build_invalid_training_example(*, seed, config):
        example = build_training_example(seed=seed, config=config)
        empty_label = tuple(
            tuple(False for _ in row) for row in example.training_label
        )
        return replace(example, training_label=empty_label)

    try:
        build_dataset_splits(
            config=config, training_example_builder=build_invalid_training_example
        )
    except DatasetInvariantError as error:
        assert "Training Label does not match Solution Path" in str(error)
    else:
        raise AssertionError("invariant failure did not stop Dataset Split generation")


def test_dataset_splits_stop_when_expected_rejection_limit_is_hit():
    config = DatasetConfig(
        width=4,
        height=4,
        maze_families=(MazeFamily.KRUSKAL,),
        split_sizes={
            DatasetSplitName.TRAIN: 1,
            DatasetSplitName.VALIDATION: 0,
            DatasetSplitName.TEST: 0,
        },
        seed_ranges={
            DatasetSplitName.TRAIN: range(0, 6),
            DatasetSplitName.VALIDATION: range(100, 101),
            DatasetSplitName.TEST: range(200, 201),
        },
        minimum_path_length=99,
        max_rejections_per_split=1,
    )

    try:
        build_dataset_splits(config=config)
    except DatasetRejectionLimitError as error:
        assert "train exceeded 1 expected rejections" in str(error)
    else:
        raise AssertionError("rejection limit did not stop Dataset Split generation")

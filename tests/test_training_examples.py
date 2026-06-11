from dreamaze.dataset import (
    MazeFamily,
    TrainingExampleConfig,
    build_training_example,
    build_kruskal_training_example,
    build_wilson_training_example,
)
from dreamaze.validation import validate_solution_mask


def test_builds_deterministic_kruskal_training_example():
    config = TrainingExampleConfig(width=16, height=16, split="train")

    example = build_kruskal_training_example(seed=20260612, config=config)
    repeated = build_kruskal_training_example(seed=20260612, config=config)

    assert example == repeated
    assert example.seed == 20260612
    assert example.split == "train"
    assert example.maze_family == MazeFamily.KRUSKAL
    assert example.metadata["maze_family"] == MazeFamily.KRUSKAL.value
    assert example.cell_graph_maze.width == 16
    assert example.cell_graph_maze.height == 16
    assert example.start_cell != example.goal_cell
    assert example.training_label == example.solution_path_mask()
    assert len(example.training_label) == len(example.maze_condition.rendered_maze)
    assert len(example.training_label[0]) == len(example.maze_condition.rendered_maze[0])
    assert example.maze_condition.rendered_maze == example.cell_graph_maze.render()
    assert example.maze_condition.start_cell == example.start_cell
    assert example.maze_condition.goal_cell == example.goal_cell
    assert example.metadata["path_length"] == len(example.solution_path)

    result = validate_solution_mask(
        grid_maze=example.maze_condition.rendered_maze,
        solution_mask=example.training_label,
        start_cell=example.rendered_start_cell,
        goal_cell=example.rendered_goal_cell,
    )
    assert result.valid is True


def test_builds_deterministic_wilson_training_example():
    config = TrainingExampleConfig(width=16, height=16, split="train")

    example = build_wilson_training_example(seed=20260612, config=config)
    repeated = build_wilson_training_example(seed=20260612, config=config)

    assert example == repeated
    assert example.seed == 20260612
    assert example.split == "train"
    assert example.maze_family == MazeFamily.WILSON
    assert example.metadata["maze_family"] == MazeFamily.WILSON.value
    assert example.cell_graph_maze.width == 16
    assert example.cell_graph_maze.height == 16
    assert example.cell_graph_maze.is_perfect()
    assert example.start_cell != example.goal_cell
    assert example.cell_graph_maze.is_border_cell(example.start_cell)
    assert example.cell_graph_maze.is_border_cell(example.goal_cell)
    assert example.training_label == example.solution_path_mask()
    assert example.solution_path == example.cell_graph_maze.unique_path(
        example.start_cell, example.goal_cell
    )
    assert len(example.training_label) == len(example.maze_condition.rendered_maze)
    assert len(example.training_label[0]) == len(example.maze_condition.rendered_maze[0])
    assert example.maze_condition.rendered_maze == example.cell_graph_maze.render()
    assert example.maze_condition.start_cell == example.start_cell
    assert example.maze_condition.goal_cell == example.goal_cell

    result = validate_solution_mask(
        grid_maze=example.maze_condition.rendered_maze,
        solution_mask=example.training_label,
        start_cell=example.rendered_start_cell,
        goal_cell=example.rendered_goal_cell,
    )
    assert result.valid is True


def test_training_example_config_selects_maze_family():
    kruskal_config = TrainingExampleConfig(
        width=16, height=16, split="train", maze_family=MazeFamily.KRUSKAL
    )
    wilson_config = TrainingExampleConfig(
        width=16, height=16, split="train", maze_family=MazeFamily.WILSON
    )

    kruskal = build_training_example(seed=41, config=kruskal_config)
    wilson = build_training_example(seed=41, config=wilson_config)
    repeated_wilson = build_training_example(seed=41, config=wilson_config)

    assert kruskal.maze_family == MazeFamily.KRUSKAL
    assert kruskal.metadata["maze_family"] == MazeFamily.KRUSKAL.value
    assert wilson.maze_family == MazeFamily.WILSON
    assert wilson.metadata["maze_family"] == MazeFamily.WILSON.value
    assert wilson == repeated_wilson
    assert kruskal != wilson


def test_kruskal_training_example_has_perfect_maze_and_unique_solution_path():
    example = build_kruskal_training_example(
        seed=17, config=TrainingExampleConfig(width=16, height=16, split="train")
    )

    assert example.cell_graph_maze.is_perfect()
    assert example.solution_path[0] == example.start_cell
    assert example.solution_path[-1] == example.goal_cell

    for first, second in zip(example.solution_path, example.solution_path[1:]):
        assert second in example.cell_graph_maze.neighbors(first)
        assert abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1

    assert example.solution_path == example.cell_graph_maze.unique_path(
        example.start_cell, example.goal_cell
    )


def test_border_endpoint_pair_uses_distinct_seed_dependent_border_cells():
    config = TrainingExampleConfig(width=16, height=16, split="train")
    first = build_kruskal_training_example(seed=3, config=config)
    second = build_kruskal_training_example(seed=4, config=config)

    for example in (first, second):
        assert example.start_cell != example.goal_cell
        assert example.cell_graph_maze.is_border_cell(example.start_cell)
        assert example.cell_graph_maze.is_border_cell(example.goal_cell)
        assert {example.start_cell, example.goal_cell} != {(0, 0), (15, 15)}

    assert (first.start_cell, first.goal_cell) != (
        second.start_cell,
        second.goal_cell,
    )


def test_maze_condition_exposes_rendered_maze_and_endpoints_without_graph_data():
    example = build_kruskal_training_example(
        seed=29, config=TrainingExampleConfig(width=16, height=16, split="validation")
    )

    assert example.maze_condition.rendered_maze == example.cell_graph_maze.render()
    assert example.maze_condition.start_cell == example.start_cell
    assert example.maze_condition.goal_cell == example.goal_cell
    assert not hasattr(example.maze_condition, "cell_graph_maze")
    assert not hasattr(example.maze_condition, "passages")

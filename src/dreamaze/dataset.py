from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from random import Random
from typing import Iterable, Mapping, Protocol, Sequence

from dreamaze.validation import validate_solution_mask

Cell = tuple[int, int]
Edge = frozenset[Cell]
BoolGrid = tuple[tuple[bool, ...], ...]


class MazeFamily(StrEnum):
    KRUSKAL = "kruskal"
    WILSON = "wilson"


class DatasetSplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetOutputFormat(StrEnum):
    NUMPY = "numpy"


class BorderEndpointPairRule(StrEnum):
    FAR_APART_BORDER_CELLS = "far_apart_border_cells"


@dataclass(frozen=True)
class DatasetConfig:
    width: int = 16
    height: int = 16
    maze_families: tuple[MazeFamily, ...] = (MazeFamily.KRUSKAL, MazeFamily.WILSON)
    split_sizes: Mapping[DatasetSplitName, int] = field(
        default_factory=lambda: {
            DatasetSplitName.TRAIN: 10_000,
            DatasetSplitName.VALIDATION: 1_000,
            DatasetSplitName.TEST: 1_000,
        }
    )
    seed_ranges: Mapping[DatasetSplitName, range] = field(
        default_factory=lambda: {
            DatasetSplitName.TRAIN: range(0, 100_000),
            DatasetSplitName.VALIDATION: range(100_000, 200_000),
            DatasetSplitName.TEST: range(200_000, 300_000),
        }
    )
    border_endpoint_pair_rule: BorderEndpointPairRule = (
        BorderEndpointPairRule.FAR_APART_BORDER_CELLS
    )
    minimum_path_length: int = 1
    output_format: DatasetOutputFormat = DatasetOutputFormat.NUMPY
    shard_size: int = 1024
    write_preview_images: bool = False
    max_rejections_per_split: int = 1000


@dataclass(frozen=True)
class TrainingExampleConfig:
    width: int = 16
    height: int = 16
    split: str = "train"
    maze_family: MazeFamily = MazeFamily.KRUSKAL


@dataclass(frozen=True)
class MazeCondition:
    rendered_maze: BoolGrid
    start_cell: Cell
    goal_cell: Cell


@dataclass(frozen=True)
class CellGraphMaze:
    width: int
    height: int
    passages: frozenset[Edge]

    def open_cell_grid(self) -> BoolGrid:
        return tuple(tuple(True for _ in range(self.width)) for _ in range(self.height))

    def render(self) -> BoolGrid:
        rendered_height = self.height * 2 + 1
        rendered_width = self.width * 2 + 1
        rendered = [
            [False for _ in range(rendered_width)] for _ in range(rendered_height)
        ]

        for row in range(self.height):
            for column in range(self.width):
                rendered[row * 2 + 1][column * 2 + 1] = True

        for edge in self.passages:
            first, second = tuple(edge)
            first_row, first_column = first
            second_row, second_column = second
            wall_row = first_row + second_row + 1
            wall_column = first_column + second_column + 1
            rendered[wall_row][wall_column] = True

        return tuple(tuple(row) for row in rendered)

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        row, column = cell
        candidates = (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
        return tuple(
            neighbor
            for neighbor in candidates
            if _in_bounds(neighbor, self.width, self.height)
            and frozenset((cell, neighbor)) in self.passages
        )

    def is_perfect(self) -> bool:
        if len(self.passages) != self.width * self.height - 1:
            return False

        start_cell = (0, 0)
        visited = set(_distances_from(self, start_cell))
        return len(visited) == self.width * self.height

    def unique_path(self, start_cell: Cell, goal_cell: Cell) -> tuple[Cell, ...]:
        return _unique_solution_path(self, start_cell, goal_cell)

    def is_border_cell(self, cell: Cell) -> bool:
        if not _in_bounds(cell, self.width, self.height):
            return False
        row, column = cell
        return row in {0, self.height - 1} or column in {0, self.width - 1}


@dataclass(frozen=True)
class TrainingExample:
    cell_graph_maze: CellGraphMaze
    start_cell: Cell
    goal_cell: Cell
    maze_family: MazeFamily
    training_label: BoolGrid
    maze_condition: MazeCondition
    split: str
    seed: int
    solution_path: tuple[Cell, ...]
    metadata: Mapping[str, int | str]

    @property
    def rendered_start_cell(self) -> Cell:
        return _rendered_cell(self.start_cell)

    @property
    def rendered_goal_cell(self) -> Cell:
        return _rendered_cell(self.goal_cell)

    def solution_path_mask(self) -> BoolGrid:
        return _solution_path_mask(
            cell_graph_maze=self.cell_graph_maze, solution_path=self.solution_path
        )


@dataclass(frozen=True)
class DatasetSplit:
    name: DatasetSplitName
    examples: tuple[TrainingExample, ...]
    rejected_seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class DatasetSplits:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


class DatasetGenerationError(RuntimeError):
    pass


class DatasetInvariantError(DatasetGenerationError):
    pass


class DatasetRejectionLimitError(DatasetGenerationError):
    pass


def build_kruskal_training_example(
    *, seed: int, config: TrainingExampleConfig
) -> TrainingExample:
    return _build_training_example(
        seed=seed, config=config, maze_family=MazeFamily.KRUSKAL
    )


def build_wilson_training_example(
    *, seed: int, config: TrainingExampleConfig
) -> TrainingExample:
    return _build_training_example(
        seed=seed, config=config, maze_family=MazeFamily.WILSON
    )


def build_training_example(
    *, seed: int, config: TrainingExampleConfig
) -> TrainingExample:
    return _build_training_example(
        seed=seed, config=config, maze_family=config.maze_family
    )


class TrainingExampleBuilder(Protocol):
    def __call__(
        self, *, seed: int, config: TrainingExampleConfig
    ) -> TrainingExample: ...


def build_dataset_splits(
    *,
    config: DatasetConfig,
    training_example_builder: TrainingExampleBuilder = build_training_example,
) -> DatasetSplits:
    _validate_dataset_config(config)
    return DatasetSplits(
        train=_build_dataset_split(
            name=DatasetSplitName.TRAIN,
            config=config,
            training_example_builder=training_example_builder,
        ),
        validation=_build_dataset_split(
            name=DatasetSplitName.VALIDATION,
            config=config,
            training_example_builder=training_example_builder,
        ),
        test=_build_dataset_split(
            name=DatasetSplitName.TEST,
            config=config,
            training_example_builder=training_example_builder,
        ),
    )


def _build_training_example(
    *, seed: int, config: TrainingExampleConfig, maze_family: MazeFamily
) -> TrainingExample:
    rng = Random(seed)
    if maze_family == MazeFamily.KRUSKAL:
        cell_graph_maze = _build_kruskal_cell_graph_maze(
            width=config.width, height=config.height, rng=rng
        )
    elif maze_family == MazeFamily.WILSON:
        cell_graph_maze = _build_wilson_cell_graph_maze(
            width=config.width, height=config.height, rng=rng
        )
    else:
        raise ValueError(f"Unsupported Maze Family: {maze_family}")

    start_cell, goal_cell = _choose_border_endpoint_pair(cell_graph_maze, rng)
    solution_path = _unique_solution_path(cell_graph_maze, start_cell, goal_cell)
    training_label = _solution_path_mask(
        cell_graph_maze=cell_graph_maze, solution_path=solution_path
    )
    maze_condition = MazeCondition(
        rendered_maze=cell_graph_maze.render(),
        start_cell=start_cell,
        goal_cell=goal_cell,
    )

    return TrainingExample(
        cell_graph_maze=cell_graph_maze,
        start_cell=start_cell,
        goal_cell=goal_cell,
        maze_family=maze_family,
        training_label=training_label,
        maze_condition=maze_condition,
        split=config.split,
        seed=seed,
        solution_path=solution_path,
        metadata={
            "width": config.width,
            "height": config.height,
            "maze_family": maze_family.value,
            "path_length": len(solution_path),
        },
    )


def _build_dataset_split(
    *,
    name: DatasetSplitName,
    config: DatasetConfig,
    training_example_builder: TrainingExampleBuilder,
) -> DatasetSplit:
    if config.split_sizes[name] == 0:
        return DatasetSplit(name=name, examples=(), rejected_seeds=())

    examples: list[TrainingExample] = []
    rejected_seeds: list[int] = []

    for seed in config.seed_ranges[name]:
        maze_family = config.maze_families[
            (len(examples) + len(rejected_seeds)) % len(config.maze_families)
        ]
        example = training_example_builder(
            seed=seed,
            config=TrainingExampleConfig(
                width=config.width,
                height=config.height,
                split=name.value,
                maze_family=maze_family,
            ),
        )
        _validate_training_example_invariants(example)

        if len(example.solution_path) < config.minimum_path_length:
            rejected_seeds.append(seed)
            if len(rejected_seeds) > config.max_rejections_per_split:
                raise DatasetRejectionLimitError(
                    f"{name.value} exceeded {config.max_rejections_per_split} "
                    "expected rejections"
                )
            continue

        examples.append(example)
        if len(examples) == config.split_sizes[name]:
            return DatasetSplit(
                name=name,
                examples=tuple(examples),
                rejected_seeds=tuple(rejected_seeds),
            )

    raise DatasetRejectionLimitError(
        f"{name.value} seed range ended before {config.split_sizes[name]} "
        "Training Examples were accepted"
    )


def _validate_dataset_config(config: DatasetConfig) -> None:
    if not config.maze_families:
        raise ValueError("Dataset config needs at least one Maze Family")
    if config.minimum_path_length < 1:
        raise ValueError("Dataset config Minimum Path Length must be positive")
    if config.max_rejections_per_split < 0:
        raise ValueError("Dataset config rejection limit cannot be negative")

    seen_seeds: set[int] = set()
    for name in DatasetSplitName:
        if name not in config.split_sizes:
            raise ValueError(f"Dataset config is missing {name.value} split size")
        if name not in config.seed_ranges:
            raise ValueError(f"Dataset config is missing {name.value} seed range")
        if config.split_sizes[name] < 0:
            raise ValueError(f"{name.value} split size cannot be negative")
        for seed in config.seed_ranges[name]:
            if seed in seen_seeds:
                raise DatasetInvariantError(
                    f"Duplicate split seed {seed} appears in multiple Dataset Splits"
                )
            seen_seeds.add(seed)


def _validate_training_example_invariants(example: TrainingExample) -> None:
    if not example.cell_graph_maze.is_perfect():
        raise DatasetInvariantError("Cell Graph Maze is not a Perfect Maze")
    if example.solution_path != example.cell_graph_maze.unique_path(
        example.start_cell, example.goal_cell
    ):
        raise DatasetInvariantError(
            "Training Example has an invalid Unique Solution Path"
        )
    if example.training_label != example.solution_path_mask():
        raise DatasetInvariantError("Training Label does not match Solution Path")
    if example.maze_condition.rendered_maze != example.cell_graph_maze.render():
        raise DatasetInvariantError(
            "Maze Condition rendering does not match Cell Graph Maze"
        )

    result = validate_solution_mask(
        grid_maze=example.maze_condition.rendered_maze,
        solution_mask=example.training_label,
        start_cell=example.rendered_start_cell,
        goal_cell=example.rendered_goal_cell,
    )
    if not result.valid:
        raise DatasetInvariantError(
            f"Training Label is not a Valid Solution: {result.reason}"
        )


def _build_kruskal_cell_graph_maze(
    *, width: int, height: int, rng: Random
) -> CellGraphMaze:
    if width < 2 or height < 2:
        raise ValueError("Kruskal Training Examples need at least a 2 by 2 maze")

    cells = [(row, column) for row in range(height) for column in range(width)]
    parent = {cell: cell for cell in cells}
    rank = {cell: 0 for cell in cells}
    edges = list(_candidate_edges(width=width, height=height))
    rng.shuffle(edges)

    passages: set[Edge] = set()
    for first, second in edges:
        if _find(parent, first) == _find(parent, second):
            continue

        _union(parent, rank, first, second)
        passages.add(frozenset((first, second)))

    return CellGraphMaze(width=width, height=height, passages=frozenset(passages))


def _build_wilson_cell_graph_maze(
    *, width: int, height: int, rng: Random
) -> CellGraphMaze:
    if width < 2 or height < 2:
        raise ValueError("Wilson Training Examples need at least a 2 by 2 maze")

    unvisited = {(row, column) for row in range(height) for column in range(width)}
    first_cell = rng.choice(sorted(unvisited))
    unvisited.remove(first_cell)
    passages: set[Edge] = set()

    while unvisited:
        start_cell = rng.choice(sorted(unvisited))
        path = [start_cell]
        path_indexes = {start_cell: 0}

        while path[-1] in unvisited:
            next_cell = rng.choice(
                _neighboring_cells(cell=path[-1], width=width, height=height)
            )
            if next_cell in path_indexes:
                loop_start = path_indexes[next_cell]
                for removed_cell in path[loop_start + 1 :]:
                    del path_indexes[removed_cell]
                path = path[: loop_start + 1]
                continue

            path_indexes[next_cell] = len(path)
            path.append(next_cell)

        for first, second in zip(path, path[1:]):
            passages.add(frozenset((first, second)))
            unvisited.discard(first)

    return CellGraphMaze(width=width, height=height, passages=frozenset(passages))


def _candidate_edges(*, width: int, height: int) -> Iterable[tuple[Cell, Cell]]:
    for row in range(height):
        for column in range(width):
            if row + 1 < height:
                yield (row, column), (row + 1, column)
            if column + 1 < width:
                yield (row, column), (row, column + 1)


def _neighboring_cells(*, cell: Cell, width: int, height: int) -> tuple[Cell, ...]:
    row, column = cell
    candidates = (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )
    return tuple(
        neighbor for neighbor in candidates if _in_bounds(neighbor, width, height)
    )


def _find(parent: dict[Cell, Cell], cell: Cell) -> Cell:
    if parent[cell] != cell:
        parent[cell] = _find(parent, parent[cell])
    return parent[cell]


def _union(
    parent: dict[Cell, Cell], rank: dict[Cell, int], first: Cell, second: Cell
) -> None:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root == second_root:
        return

    if rank[first_root] < rank[second_root]:
        parent[first_root] = second_root
    elif rank[first_root] > rank[second_root]:
        parent[second_root] = first_root
    else:
        parent[second_root] = first_root
        rank[first_root] += 1


def _choose_border_endpoint_pair(
    cell_graph_maze: CellGraphMaze, rng: Random
) -> tuple[Cell, Cell]:
    border_cells = _border_cells(
        width=cell_graph_maze.width, height=cell_graph_maze.height
    )
    best_pairs: list[tuple[Cell, Cell]] = []
    best_distance = -1

    for index, first in enumerate(border_cells):
        distances = _distances_from(cell_graph_maze, first)
        for second in border_cells[index + 1 :]:
            distance = distances[second]
            if distance > best_distance:
                best_distance = distance
                best_pairs = [(first, second)]
            elif distance == best_distance:
                best_pairs.append((first, second))

    return rng.choice(best_pairs)


def _border_cells(*, width: int, height: int) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for row in range(height):
        for column in range(width):
            if row in {0, height - 1} or column in {0, width - 1}:
                cells.append((row, column))
    return tuple(cells)


def _distances_from(cell_graph_maze: CellGraphMaze, start_cell: Cell) -> dict[Cell, int]:
    distances = {start_cell: 0}
    frontier = deque([start_cell])

    while frontier:
        cell = frontier.popleft()
        for neighbor in cell_graph_maze.neighbors(cell):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[cell] + 1
            frontier.append(neighbor)

    return distances


def _unique_solution_path(
    cell_graph_maze: CellGraphMaze, start_cell: Cell, goal_cell: Cell
) -> tuple[Cell, ...]:
    previous: dict[Cell, Cell | None] = {start_cell: None}
    frontier = deque([start_cell])

    while frontier:
        cell = frontier.popleft()
        if cell == goal_cell:
            break
        for neighbor in cell_graph_maze.neighbors(cell):
            if neighbor in previous:
                continue
            previous[neighbor] = cell
            frontier.append(neighbor)

    path = [goal_cell]
    while path[-1] != start_cell:
        predecessor = previous[path[-1]]
        if predecessor is None:
            break
        path.append(predecessor)
    path.reverse()

    return tuple(path)


def _solution_path_mask(
    *, cell_graph_maze: CellGraphMaze, solution_path: Sequence[Cell]
) -> BoolGrid:
    rendered = [
        [False for _ in range(cell_graph_maze.width * 2 + 1)]
        for _ in range(cell_graph_maze.height * 2 + 1)
    ]

    for cell in solution_path:
        row, column = _rendered_cell(cell)
        rendered[row][column] = True

    for first, second in zip(solution_path, solution_path[1:]):
        first_row, first_column = first
        second_row, second_column = second
        rendered[first_row + second_row + 1][first_column + second_column + 1] = True

    return tuple(tuple(row) for row in rendered)


def _rendered_cell(cell: Cell) -> Cell:
    row, column = cell
    return row * 2 + 1, column * 2 + 1


def _in_bounds(cell: Cell, width: int, height: int) -> bool:
    row, column = cell
    return 0 <= row < height and 0 <= column < width

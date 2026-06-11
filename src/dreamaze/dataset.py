import hashlib
import json
import struct
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Protocol, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

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


@dataclass(frozen=True)
class DatasetArtifactShard:
    split: DatasetSplitName
    name: str
    example_count: int
    seeds: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class DatasetArtifactManifest:
    status: str
    config: Mapping[str, Any]
    split_names: tuple[str, ...]
    split_counts: Mapping[str, int]
    seeds: Mapping[str, tuple[int, ...]]
    shards: tuple[DatasetArtifactShard, ...]


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


def write_dataset_artifacts(
    *,
    config: DatasetConfig,
    output_dir: str | Path,
    training_example_builder: TrainingExampleBuilder = build_training_example,
) -> DatasetArtifactManifest:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    completed_manifest = _load_completed_manifest_if_reusable(
        path=output_path / "manifest.json", config=config, output_dir=output_path
    )
    if completed_manifest is not None:
        return completed_manifest

    splits = build_dataset_splits(
        config=config, training_example_builder=training_example_builder
    )
    if config.write_preview_images:
        _write_preview_images(output_dir=output_path / "previews", splits=splits)

    artifact_shards: list[DatasetArtifactShard] = []
    for split in (splits.train, splits.validation, splits.test):
        for shard_index, examples in enumerate(
            _chunk_training_examples(split.examples, config.shard_size)
        ):
            shard_name = f"{split.name.value}-{shard_index:05d}.npz"
            shard_path = output_path / shard_name
            _write_dataset_artifact_shard(path=shard_path, examples=examples)
            artifact_shards.append(
                DatasetArtifactShard(
                    split=split.name,
                    name=shard_name,
                    example_count=len(examples),
                    seeds=tuple(example.seed for example in examples),
                    sha256=_sha256_file(shard_path),
                )
            )

    manifest = DatasetArtifactManifest(
        status="complete",
        config=_dataset_config_manifest(config),
        split_names=tuple(split.value for split in DatasetSplitName),
        split_counts={
            splits.train.name.value: len(splits.train.examples),
            splits.validation.name.value: len(splits.validation.examples),
            splits.test.name.value: len(splits.test.examples),
        },
        seeds={
            splits.train.name.value: tuple(
                example.seed for example in splits.train.examples
            ),
            splits.validation.name.value: tuple(
                example.seed for example in splits.validation.examples
            ),
            splits.test.name.value: tuple(
                example.seed for example in splits.test.examples
            ),
        },
        shards=tuple(artifact_shards),
    )
    _write_manifest(output_path / "manifest.json", manifest)
    return manifest


def load_dataset_artifact_shard(path: str | Path) -> Mapping[str, Any]:
    with ZipFile(Path(path), "r") as archive:
        return json.loads(archive.read("arrays.json").decode("utf-8"))


def load_dataset_artifact_manifest(path: str | Path) -> DatasetArtifactManifest:
    payload = json.loads(Path(path).read_text())
    return DatasetArtifactManifest(
        status=payload["status"],
        config=payload["config"],
        split_names=tuple(payload["split_names"]),
        split_counts=payload["split_counts"],
        seeds={split: tuple(seeds) for split, seeds in payload["seeds"].items()},
        shards=tuple(
            DatasetArtifactShard(
                split=DatasetSplitName(shard["split"]),
                name=shard["name"],
                example_count=shard["example_count"],
                seeds=tuple(shard["seeds"]),
                sha256=shard["sha256"],
            )
            for shard in payload["shards"]
        ),
    )


def _chunk_training_examples(
    examples: Sequence[TrainingExample], shard_size: int
) -> Iterable[tuple[TrainingExample, ...]]:
    if shard_size < 1:
        raise ValueError("Dataset Artifact shard size must be positive")

    for start in range(0, len(examples), shard_size):
        yield tuple(examples[start : start + shard_size])


def _load_completed_manifest_if_reusable(
    *, path: Path, config: DatasetConfig, output_dir: Path
) -> DatasetArtifactManifest | None:
    if not path.exists():
        return None

    try:
        manifest_payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DatasetGenerationError(
            f"Cannot resume Dataset Artifact build from invalid manifest: {path}"
        ) from error

    status = manifest_payload.get("status")
    if status != "complete":
        raise DatasetGenerationError(
            f"Found incomplete Dataset Artifact build in {path}; remove it or "
            "resume from a complete manifest"
        )

    manifest = load_dataset_artifact_manifest(path)
    if manifest.config != _dataset_config_manifest(config):
        return None

    for shard in manifest.shards:
        shard_path = output_dir / shard.name
        if not shard_path.exists():
            raise DatasetGenerationError(
                f"Completed manifest references missing Dataset Artifact shard: "
                f"{shard.name}"
            )
        if _sha256_file(shard_path) != shard.sha256:
            raise DatasetGenerationError(
                f"Completed manifest integrity check failed for Dataset Artifact "
                f"shard: {shard.name}"
            )

    return manifest


def _write_dataset_artifact_shard(
    *, path: Path, examples: Sequence[TrainingExample]
) -> None:
    payload = _dataset_artifact_payload(examples)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "arrays.json",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        archive.writestr(
            "maze_condition.npy",
            _npy_bytes(payload["maze_condition"], descr="|u1"),
        )
        archive.writestr(
            "solution_mask.npy",
            _npy_bytes(payload["solution_mask"], descr="|u1"),
        )
        archive.writestr(
            "start_cell.npy",
            _npy_bytes(payload["start_cell"], descr="<i8"),
        )
        archive.writestr(
            "goal_cell.npy",
            _npy_bytes(payload["goal_cell"], descr="<i8"),
        )
        archive.writestr("seed.npy", _npy_bytes(payload["seed"], descr="<i8"))


def _write_preview_images(*, output_dir: Path, splits: DatasetSplits) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in (splits.train, splits.validation, splits.test):
        for example in split.examples:
            preview_path = output_dir / f"{split.name.value}-{example.seed:06d}.pgm"
            preview_path.write_text(_preview_image_pgm(example))


def _preview_image_pgm(example: TrainingExample) -> str:
    rendered_maze = example.maze_condition.rendered_maze
    solution_mask = example.training_label
    rows = len(rendered_maze)
    columns = len(rendered_maze[0])
    pixels: list[str] = []

    for row_index, row in enumerate(rendered_maze):
        values: list[str] = []
        for column_index, is_open in enumerate(row):
            if solution_mask[row_index][column_index]:
                values.append("180")
            elif is_open:
                values.append("255")
            else:
                values.append("0")
        pixels.append(" ".join(values))

    return f"P2\n{columns} {rows}\n255\n" + "\n".join(pixels) + "\n"


def _dataset_artifact_payload(examples: Sequence[TrainingExample]) -> Mapping[str, Any]:
    return {
        "maze_condition": [
            _bool_grid_to_array(example.maze_condition.rendered_maze)
            for example in examples
        ],
        "solution_mask": [
            _bool_grid_to_array(example.training_label) for example in examples
        ],
        "start_cell": [list(example.start_cell) for example in examples],
        "goal_cell": [list(example.goal_cell) for example in examples],
        "maze_family": [example.maze_family.value for example in examples],
        "split": [example.split for example in examples],
        "seed": [example.seed for example in examples],
        "metadata": _metadata_arrays(examples),
    }


def _metadata_arrays(examples: Sequence[TrainingExample]) -> Mapping[str, list[Any]]:
    keys = sorted({key for example in examples for key in example.metadata})
    return {key: [example.metadata.get(key) for example in examples] for key in keys}


def _bool_grid_to_array(grid: BoolGrid) -> list[list[int]]:
    return [[1 if cell else 0 for cell in row] for row in grid]


def _npy_bytes(array: Any, *, descr: str) -> bytes:
    shape = _array_shape(array)
    flattened = list(_flatten_array(array))
    header = _npy_header(descr=descr, shape=shape)

    if descr == "|u1":
        data = bytes(flattened)
    elif descr == "<i8":
        data = b"".join(struct.pack("<q", value) for value in flattened)
    else:
        raise ValueError(f"Unsupported Dataset Artifact array dtype: {descr}")

    return header + data


def _npy_header(*, descr: str, shape: tuple[int, ...]) -> bytes:
    shape_repr = f"({shape[0]},)" if len(shape) == 1 else repr(shape)
    header = (
        f"{{'descr': '{descr}', 'fortran_order': False, "
        f"'shape': {shape_repr}, }}"
    )
    header_length = len(header) + 1
    padding = 16 - ((10 + header_length) % 16)
    header_bytes = (header + (" " * padding) + "\n").encode("ascii")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes


def _array_shape(array: Any) -> tuple[int, ...]:
    if not isinstance(array, list):
        return ()
    if not array:
        return (0,)
    return (len(array),) + _array_shape(array[0])


def _flatten_array(array: Any) -> Iterable[int]:
    if isinstance(array, list):
        for item in array:
            yield from _flatten_array(item)
    else:
        yield int(array)


def _write_manifest(path: Path, manifest: DatasetArtifactManifest) -> None:
    manifest_payload = {
        "status": manifest.status,
        "config": manifest.config,
        "split_names": list(manifest.split_names),
        "split_counts": dict(manifest.split_counts),
        "seeds": {split: list(seeds) for split, seeds in manifest.seeds.items()},
        "shards": [
            {
                "split": shard.split.value,
                "name": shard.name,
                "example_count": shard.example_count,
                "seeds": list(shard.seeds),
                "sha256": shard.sha256,
            }
            for shard in manifest.shards
        ],
    }
    path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")


def _dataset_config_manifest(config: DatasetConfig) -> Mapping[str, Any]:
    return {
        "width": config.width,
        "height": config.height,
        "maze_families": [maze_family.value for maze_family in config.maze_families],
        "split_sizes": {
            split.value: config.split_sizes[split] for split in DatasetSplitName
        },
        "seed_ranges": {
            split.value: {
                "start": config.seed_ranges[split].start,
                "stop": config.seed_ranges[split].stop,
                "step": config.seed_ranges[split].step,
            }
            for split in DatasetSplitName
        },
        "border_endpoint_pair_rule": config.border_endpoint_pair_rule.value,
        "minimum_path_length": config.minimum_path_length,
        "output_format": config.output_format.value,
        "shard_size": config.shard_size,
        "write_preview_images": config.write_preview_images,
        "max_rejections_per_split": config.max_rejections_per_split,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

# Dreamaze

Dreamaze is an experiment to teach a small AI model to solve mazes.

The first version is intentionally simple: it uses clean grid mazes, asks the model to draw the correct path from start to finish, and then checks whether that path is truly valid.

## What

Dreamaze will build a maze-solving model that:

- receives a maze with a marked start and goal
- predicts the path through the maze
- returns that path as a simple mask
- checks whether the path really reaches the goal without crossing walls

The first mazes are 16 by 16 cell grid mazes made with Kruskal's and Wilson's maze-generation methods.

## Why

The goal is to prove that the model is actually solving the maze, not just making a picture that looks right.

That means:

- the model must produce the answer path itself
- normal pathfinding algorithms can help create training answers
- normal pathfinding algorithms cannot secretly solve the maze during the demo
- success is measured by how often the model gives a real valid path

The first target is for the model to solve at least 80% of new test mazes in one try.

## How

The project will be built in stages.

1. Build a dataset generator.
   It creates perfect mazes, chooses start and goal cells, finds the correct answer path for training, and saves the data.

2. Train a small diffusion model.
   The model learns from the generated mazes and answer paths.

3. Measure the model.
   A checker verifies whether each predicted path is connected, stays inside open maze cells, reaches the goal, and has no extra branches.

4. Build a proof demo.
   The demo shows a maze, the model's predicted path, and whether the prediction is valid.

Training and the proof demo are planned for Hugging Face. Training should run on Hugging Face compute, and the demo should run on Hugging Face Spaces.

## Current Implementation

Dreamaze now includes Graph Validation for proposed Solution Masks, Dataset
Builder support for deterministic Kruskal and Wilson Training Examples plus
deterministic train, validation, and test Dataset Splits, a small Conditional
Diffusion Solver implemented with Hugging Face Diffusers, evaluation, Hugging
Face Jobs launch packaging, and a Hugging Face Spaces Proof Demo target. The
Dataset Builder can persist those splits as sharded Dataset Artifacts with a
resumable manifest.

The Proof Demo has been updated with a fully automated, intuitive user
experience: visitors see only a single prominent "Solve New Maze" button and
the result. All previous configuration (maze family, seed, sampling steps,
retries, debug reveal) is chosen internally with sensible defaults. The result
area now features a rich self-contained real-time visualization (built with
embedded HTML, CSS, and JavaScript) that plays back the Conditional Diffusion
Solver's full denoising trajectory step-by-step. This lets users literally
watch the diffusion process solve the maze from initial noise to the final
Solution Mask before Graph Validation is applied. New supporting library
APIs (`sample_conditional_diffusion_solution_mask_trajectory` and
`build_diffusion_viz_html`) power the animated player while the core Runtime
Solving and strict validation logic remain unchanged.

The validator checks a submitted mask without creating, filling, repairing, or replacing the path. It accepts a mask only when the marked cells form one continuous 4-Way Movement route from the Start Cell to the Goal Cell through open Grid Maze cells.

Invalid masks return a structured Validation Reason:

- `empty_mask`
- `missing_start`
- `missing_goal`
- `wall_crossing`
- `disconnected`
- `diagonal_only`
- `extra_branch`

The Training Example builder creates a 16 by 16 Cell Graph Maze from a fixed
seed, explicit Maze Family, and config. It currently supports Kruskal and Wilson
Maze Families, chooses a seed-dependent Border Endpoint Pair, computes the
Unique Solution Path as the Training Label, renders the Maze Condition, and
checks cleanly through Solution Validation. The Training Label uses the same
rendered coordinate system as the Rendered Maze so wall-separated neighboring
cells do not look connected to Graph Validation.

Use `build_training_example(...)` with `TrainingExampleConfig.maze_family` to
select a Maze Family, or call the family-specific helpers
`build_kruskal_training_example(...)` and `build_wilson_training_example(...)`.

Dataset Splits are generated with `build_dataset_splits(...)` and
`DatasetConfig`. The config carries the First Maze Size, Maze Families, train /
validation / test split sizes, separate fixed seed ranges, the Border Endpoint
Pair rule, Minimum Path Length, output format, shard size, and Preview Image
setting. Each split uses its own fixed seed range, so generated Training
Examples are deterministic and split membership is stable. Expected Minimum
Path Length rejections are skipped and replaced until the requested split is
full or the rejection limit is hit. Invariant failures, including duplicate
split seeds, invalid Unique Solution Paths, invalid Training Labels, or
rendering mismatches, stop generation loudly.

Dataset Artifacts are written with `write_dataset_artifacts(...)`. The writer
stores compressed split shards, each containing the Maze Condition arrays,
Solution Masks, endpoint positions, Maze Family, split, seed, and metadata
needed for training and audit. It also writes `manifest.json` with the Dataset
Config, split names, split counts, seeds used, shard names, shard counts,
SHA-256 integrity checks, and build status. Re-running against a complete
matching manifest skips regeneration after checking every shard hash; incomplete
manifests or missing shards fail clearly. Optional Preview Images are written
under `previews/` for inspection and are not part of the training artifact
source data.

The Dataset Builder CLI can run the same artifact writer for a small smoke-test
dataset or the First Dataset Size:

```bash
dreamaze-dataset build --preset tiny --output-dir ./artifacts/tiny --preview-images
dreamaze-dataset build --preset first --output-dir ./artifacts/first
```

The `tiny` preset writes 3 training, 1 validation, and 1 test Training Example
with small 4 by 4 Grid Mazes for CI and local checks. The `first` preset uses
the First Dataset Size: 10,000 training, 1,000 validation, and 1,000 test
Training Examples. Re-running the command against a completed matching manifest
resumes by reusing the existing Dataset Artifacts after the writer verifies
shard integrity.

The training path uses a custom Conditional Diffusion Solver implemented with
Hugging Face Diffusers `UNet2DModel`. It loads sharded Dataset Artifacts,
conditions on Maze Condition arrays, trains from scratch toward Solution Masks,
and writes Diffusers checkpoint directories containing the UNet, scheduler, and
Dreamaze metadata. This is the single Runtime Solving model path; Dreamaze does
not use Stable Diffusion or fixture weights for the Proof Demo.

Create a training config such as:

```json
{
  "dataset_dir": "./artifacts/tiny",
  "checkpoint_dir": "./artifacts/checkpoints",
  "split": "train",
  "batch_size": 2,
  "sampling_steps": 3,
  "max_train_steps": 1,
  "checkpoint_every_steps": 1,
  "learning_rate": 0.05,
  "seed": 123,
  "device": "cpu",
  "precision": "float32",
  "num_workers": 0
}
```

Then run:

```bash
dreamaze-train --config ./training.json
```

Evaluate a checkpoint against unseen validation or test Dataset Artifacts with
a config such as:

```json
{
  "dataset_dir": "./artifacts/tiny",
  "checkpoint_path": "./artifacts/checkpoints/checkpoint-step-000001",
  "split": "validation",
  "sampling_steps": 3,
  "retry_count": 1,
  "seed": 456,
  "report_path": "./artifacts/evaluation.json",
  "device": "cpu",
  "precision": "float32"
}
```

Then run:

```bash
dreamaze-evaluate --config ./evaluation.json
```

The evaluation report records the Dataset Split, checkpoint identity, sampling
settings, Single-Sample Success as the official Valid-Solution Rate, optional
Retry Success marked as excluded from the official score, mask overlap, and
Validation Reason failure counts. Evaluation samples Solution Masks from the
checkpoint and validates them; it does not use a classical solver to repair or
replace Runtime Solving.

## Hugging Face Jobs

Training and evaluation can be launched through Hugging Face Jobs without
assuming the user's local GPU is available. The local CLI builds an `hf jobs uv
run` command around a remote-ready UV script, and `--dry-run` prints the command
without starting paid compute:

```json
{
  "hardware_flavor": "cpu-basic",
  "timeout": "10m",
  "python": "3.12",
  "dataset_preset": "tiny",
  "dataset_dir": "/tmp/dreamaze/dataset",
  "checkpoint_dir": "/tmp/dreamaze/checkpoints",
  "evaluation_output": "/tmp/dreamaze/evaluation.json",
  "batch_size": 1,
  "sampling_steps": 2,
  "max_train_steps": 1,
  "checkpoint_every_steps": 1,
  "learning_rate": 0.01,
  "training_seed": 7,
  "evaluation_seed": 11,
  "retry_count": 1,
  "eval_split": "validation",
  "device": "cpu",
  "precision": "float32",
  "num_workers": 0
}
```

```bash
dreamaze-hf-job --config ./hf-job.json --dry-run
```

Remove `--dry-run` to submit the job with the installed `hf` CLI. Authenticate
first with `hf auth login`; artifact upload requires a token with write access.
When `"output_repo"` is set, the launch command passes `HF_TOKEN` as a Hugging
Face Jobs secret and the remote script uploads the run directory:

```json
{
  "output_repo": "your-name/dreamaze-artifacts",
  "output_repo_type": "dataset",
  "output_path_in_repo": "runs/tiny-smoke"
}
```

Start with `hardware_flavor` set to `cpu-basic`, `dataset_preset` set to
`tiny`, and `max_train_steps` set to `1`. Scale to GPU flavors such as
`t4-small`, larger batch sizes, more sampling steps, longer checkpoint cadence,
and the `first` Dataset Artifact preset only after the dry-run command and tiny
remote smoke job behave as expected.

For the budget GPU path that fits the current single-process trainer, use the
`best_gpu` compute profile. It resolves to the `t4-small` Hugging Face Jobs
hardware flavor, CUDA device settings, float16 precision, the First Dataset
Size preset, and conservative training defaults intended to stay inside a small
prepaid-credit budget:

```json
{
  "compute_profile": "best_gpu",
  "output_repo": "your-name/dreamaze-artifacts",
  "output_repo_type": "dataset",
  "output_path_in_repo": "runs/best-gpu"
}
```

```bash
dreamaze-hf-job --config ./hf-best-gpu.json --dry-run
```

Keep `--dry-run` until the command, output repo, and expected spend are checked.

## Hugging Face Spaces Proof Demo

The Proof Demo target lives in `spaces/proof_demo`. It is a Gradio Space that
calls the same Conditional Diffusion Solver sampling path used by evaluation,
then runs Solution Validation against the generated Solution Mask. Invalid
outputs are displayed as invalid with their Validation Reason; the Space does
not use DFS, BFS, A*, or any other classical pathfinding fallback to repair or
replace Runtime Solving.

The public demo UI has been deliberately simplified and automated per user
request: end users see **only a single prominent "Solve New Maze" (or "Play")
button and the result area**. All configuration is handled internally with
good defaults (e.g. 32 sampling steps for visible animation length, fresh
varied Grid Mazes on every invocation via randomized seeds within safe ranges,
single-sample execution for the official score, debug reveal disabled). This
keeps the experience extremely intuitive while still exercising the full
learned solver on live Runtime Solving.

The result is a rich, self-contained visualization built with embedded
HTML/CSS/JavaScript. It renders the input Rendered Maze (with Start Cell in
green and Goal Cell in red) and then auto-plays a smooth animation of the
Conditional Diffusion Solver's complete denoising trajectory. Viewers watch
the Solution Mask emerge in real time from pure noise through iterative
refinement steps until the final mask is produced. A step counter, replay
control, legend, and prominent verdict ("Valid Solution" or "Invalid Solution"
plus the exact Validation Reason) are included. The animation uses the exact
intermediate states from the solver (exposed via the new
`sample_conditional_diffusion_solution_mask_trajectory` helper) so it
faithfully represents the diffusion process rather than a post-hoc effect.

The Space requires a trained Dreamaze Diffusers checkpoint directory:

```bash
DREAMAZE_CHECKPOINT_PATH=/path/to/checkpoint-step-000001
```

If that variable is unset or points to a missing checkpoint, the Space fails at
startup. The public demo is considered deployable only when a Trained Solver
Checkpoint is configured.

The first public deployment is:

- Proof Demo Space: <https://huggingface.co/spaces/Srini410/dreamaze-proof-demo>
- Artifact dataset repo: <https://huggingface.co/datasets/Srini410/dreamaze-artifacts>
- Solver model repo: <https://huggingface.co/Srini410/dreamaze-solver>

The Proof Demo Space runs on Hugging Face ZeroGPU because the deployment target
is a public learned-solver demo.

The earlier public Space was configured with the initial trained tracer-bullet
checkpoint:

- Checkpoint: <https://huggingface.co/Srini410/dreamaze-solver/blob/main/checkpoints/checkpoint-step-000020.json>
- Evaluation report: <https://huggingface.co/datasets/Srini410/dreamaze-artifacts/blob/main/runs/tiny-trained-step-000020/evaluation.json>

That checkpoint proved the deployment path could load a trained solver, but it
does not meet the First Success Target. Its tiny validation run reported 0.0
Single-Sample Success. The current model path is the Diffusers checkpoint
directory format described above.

To create the deployment repos with the installed `hf` CLI:

```bash
hf repo create Srini410/dreamaze-proof-demo --repo-type space --space_sdk gradio --exist-ok
hf repo create Srini410/dreamaze-artifacts --repo-type dataset --private --exist-ok
hf repo create Srini410/dreamaze-solver --exist-ok
```

Then upload the Space app and package source:

```bash
hf upload Srini410/dreamaze-proof-demo spaces/proof_demo/app.py app.py --repo-type space
hf upload Srini410/dreamaze-proof-demo spaces/proof_demo/README.md README.md --repo-type space
hf upload Srini410/dreamaze-proof-demo spaces/proof_demo/requirements.txt requirements.txt --repo-type space
hf upload Srini410/dreamaze-proof-demo src src --repo-type space
```

Set the Space hardware to ZeroGPU with the Hub API:

```bash
python -c "from huggingface_hub import HfApi, SpaceHardware; HfApi().request_space_hardware('Srini410/dreamaze-proof-demo', SpaceHardware.ZERO_A10G)"
```

Use `zero-a10g` when the account is eligible for ZeroGPU and queued/free GPU
execution is preferable to predictable latency. Move to paid dedicated GPU
hardware only when the trained checkpoint needs lower latency or more reliable
capacity, because the Space owner is billed while that hardware is attached.

Manual browser smoke checks (fully automated Proof Demo UI):

- Load the public Space (https://huggingface.co/spaces/Srini410/dreamaze-proof-demo). You should see only a prominent "Solve New Maze" button plus an initial result (no configuration controls, sliders, dropdowns, or checkboxes are exposed to the visitor).
- Click the button and confirm a fresh Grid Maze appears together with an animated real-time playback of the Conditional Diffusion Solver's denoising trajectory (the Solution Mask claim evolving from noise over the refinement steps via embedded HTML/CSS/JS). The animation starts automatically at a comfortable visible pace.
- Confirm the animation ends on a final mask, a clear verdict using the project's exact terms ("Valid Solution" / "Invalid Solution"), and the precise Validation Reason when applicable. Everything is produced by pure model sampling + Graph Validation with no classical solver involved.
- Click again (or refresh): a different maze + independent single-sample solve should appear. Internal automation selects Maze Family and seed for variety on every invocation while keeping the official result as Single-Sample Success.

New library surface area supporting the demo (exported from the package):

- `sample_conditional_diffusion_solution_mask_trajectory(...)` — returns the full list of intermediate masks from the custom reverse diffusion process (initial noise through final clean mask).
- `build_diffusion_viz_html(result, ...)` — given a `ProofDemoResult` that captured the trajectory (plus optional family/seed metadata), returns a complete self-contained HTML block with CSS and JS that renders and animates the solving process.

Run the test suite with:

```bash
pytest
```

## Not Yet

These ideas are intentionally saved for later:

- drawing a squiggle to create a maze
- illustrated fantasy-style maze art
- large mazes
- impossible mazes with no solution
- polished mobile app experience

The first job is smaller and stricter: prove that the model can solve clean mazes.

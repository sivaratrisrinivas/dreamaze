# Dreamaze

Dreamaze trains a small Conditional Diffusion Solver to solve grid mazes.

## What

Dreamaze builds deterministic maze datasets, trains a Diffusers `UNet2DModel`
to predict a Solution Mask from a Rendered Maze plus Start Cell and Goal Cell,
and validates the model's mask with strict Graph Validation.

The public Proof Demo is a Hugging Face Docker Space:

- App: https://srini410-dreamaze-proof-demo.hf.space
- Space repo: https://huggingface.co/spaces/Srini410/dreamaze-proof-demo

The demo shows one button. Clicking it streams the model's denoising steps from
the backend with Server-Sent Events, so the browser displays the maze
transformation as Runtime Solving happens.

## Why

The goal is to prove whether the learned model can actually solve unseen mazes.

Runtime Solving must come from the trained Dreamaze Conditional Diffusion Solver.
Classical DFS/BFS/A* pathfinding may create Training Labels and validate Dataset
Builder invariants, but it must not repair, replace, or secretly generate demo
or evaluation outputs.

The official score is Single-Sample Success: one model sample is valid only if
the predicted mask forms one continuous 4-Way Movement route from Start Cell to
Goal Cell through open maze cells.

## How

Install for local development:

```bash
python -m pip install -e .
```

Run tests:

```bash
pytest -q
```

Build Dataset Artifacts:

```bash
dreamaze-dataset build --preset tiny --output-dir ./artifacts/tiny --preview-images
dreamaze-dataset build --preset first --output-dir ./artifacts/first
dreamaze-dataset build --preset larger --output-dir ./artifacts/larger
```

Train with a JSON config:

```bash
dreamaze-train --config ./training.json
```

Evaluate a checkpoint:

```bash
dreamaze-evaluate --config ./evaluation.json
```

Launch a Hugging Face Job:

```bash
dreamaze-hf-job --config ./hf-job.json --dry-run
dreamaze-hf-job --config ./hf-job.json
dreamaze-hf-job --config ./configs/hf-job-larger-gpu.json --dry-run
```

`configs/hf-job-larger-gpu.json` uses the `larger` Dataset Builder preset and
a bounded `t4-small` GPU run. Deploy its checkpoint only after evaluation shows
an improved Single-Sample Success result; do not treat retry-assisted success as
the official score.

Deploy the Proof Demo Space:

```bash
DREAMAZE_CHECKPOINT_REPO_ID=Srini410/dreamaze-solver \
DREAMAZE_CHECKPOINT_REPO_PATH=checkpoints/diffusers-t4-budget-1000-retry-20260612b/checkpoint-step-001000 \
./run.sh
```

`./run.sh` uploads the Docker Space, configures checkpoint variables, requests
`T4_SMALL` hardware, sets a 300-second idle sleep timeout, and restarts the
Space. Checkpoint loading is lazy: the Space starts first, then downloads the
solver checkpoint when `Solve New Maze` is clicked.

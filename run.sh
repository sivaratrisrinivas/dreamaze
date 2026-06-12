#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SPACE_ID="${SPACE_ID:-Srini410/dreamaze-proof-demo}"
SPACE_HARDWARE="${SPACE_HARDWARE:-T4_SMALL}"
SPACE_SLEEP_TIME="${SPACE_SLEEP_TIME:-300}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install/login once outside this command, then rerun ./run.sh." >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "You are not logged in to Hugging Face. Run 'hf auth login' once, then rerun ./run.sh." >&2
  exit 1
fi

if [ -z "${DREAMAZE_CHECKPOINT_PATH:-}" ] && {
  [ -z "${DREAMAZE_CHECKPOINT_REPO_ID:-}" ] || [ -z "${DREAMAZE_CHECKPOINT_REPO_PATH:-}" ]
}; then
  cat >&2 <<'MSG'
Set either:
  1. DREAMAZE_CHECKPOINT_REPO_ID and DREAMAZE_CHECKPOINT_REPO_PATH for a checkpoint stored on Hugging Face, or
  2. DREAMAZE_CHECKPOINT_PATH for a checkpoint path that already exists inside the Space.

This command does not download models or libraries locally. It deploys Dreamaze to Hugging Face,
where the Docker build installs dependencies, checkpoint loading happens, and GPU runtime solving happens.

Example:
  DREAMAZE_CHECKPOINT_REPO_ID=Srini410/dreamaze-solver DREAMAZE_CHECKPOINT_REPO_PATH=checkpoints/diffusers-t4-budget-1000/checkpoint-step-001000 ./run.sh
MSG
  exit 1
fi

echo "Deploying Dreamaze Proof Demo to Hugging Face Space: $SPACE_ID"
echo "No local Python packages or models will be installed/downloaded."

hf repo create "$SPACE_ID" --repo-type space --space_sdk docker --exist-ok

hf upload "$SPACE_ID" spaces/proof_demo/app.py app.py --repo-type space --commit-message "Deploy Dreamaze HTML proof demo"
hf upload "$SPACE_ID" spaces/proof_demo/README.md README.md --repo-type space --commit-message "Update Dreamaze Space README"
hf upload "$SPACE_ID" spaces/proof_demo/requirements.txt requirements.txt --repo-type space --commit-message "Update Dreamaze Space dependencies"
hf upload "$SPACE_ID" spaces/proof_demo/Dockerfile Dockerfile --repo-type space --commit-message "Update Dreamaze Docker runtime"
hf upload "$SPACE_ID" spaces/proof_demo/static static --repo-type space --commit-message "Deploy Dreamaze static UI"
hf upload "$SPACE_ID" src src --repo-type space --commit-message "Deploy Dreamaze package source"

SPACE_ID="$SPACE_ID" SPACE_HARDWARE="$SPACE_HARDWARE" SPACE_SLEEP_TIME="$SPACE_SLEEP_TIME" python - <<'PY'
import os
from huggingface_hub import HfApi, SpaceHardware

space_id = os.environ["SPACE_ID"]
checkpoint_path = os.environ.get("DREAMAZE_CHECKPOINT_PATH")
checkpoint_repo_id = os.environ.get("DREAMAZE_CHECKPOINT_REPO_ID")
checkpoint_repo_path = os.environ.get("DREAMAZE_CHECKPOINT_REPO_PATH")
checkpoint_revision = os.environ.get("DREAMAZE_CHECKPOINT_REVISION")
hardware_name = os.environ["SPACE_HARDWARE"]

api = HfApi()


def delete_variable_if_present(key: str) -> None:
    try:
        api.delete_space_variable(repo_id=space_id, key=key)
    except Exception:
        pass


if checkpoint_path:
    api.add_space_variable(repo_id=space_id, key="DREAMAZE_CHECKPOINT_PATH", value=checkpoint_path)
else:
    delete_variable_if_present("DREAMAZE_CHECKPOINT_PATH")
if checkpoint_repo_id and checkpoint_repo_path:
    api.add_space_variable(repo_id=space_id, key="DREAMAZE_CHECKPOINT_REPO_ID", value=checkpoint_repo_id)
    api.add_space_variable(repo_id=space_id, key="DREAMAZE_CHECKPOINT_REPO_PATH", value=checkpoint_repo_path)
else:
    delete_variable_if_present("DREAMAZE_CHECKPOINT_REPO_ID")
    delete_variable_if_present("DREAMAZE_CHECKPOINT_REPO_PATH")
if checkpoint_revision:
    api.add_space_variable(repo_id=space_id, key="DREAMAZE_CHECKPOINT_REVISION", value=checkpoint_revision)
else:
    delete_variable_if_present("DREAMAZE_CHECKPOINT_REVISION")
sleep_time = int(os.environ["SPACE_SLEEP_TIME"])
api.request_space_hardware(
    repo_id=space_id,
    hardware=SpaceHardware[hardware_name],
    sleep_time=sleep_time,
)
api.restart_space(repo_id=space_id)
if checkpoint_path:
    print(f"Configured DREAMAZE_CHECKPOINT_PATH={checkpoint_path}")
else:
    print(f"Configured DREAMAZE_CHECKPOINT_REPO_ID={checkpoint_repo_id}")
    print(f"Configured DREAMAZE_CHECKPOINT_REPO_PATH={checkpoint_repo_path}")
print(f"Requested hardware={hardware_name}")
print(f"Configured sleep_time={sleep_time} seconds")
PY

echo "Space repo: https://huggingface.co/spaces/$SPACE_ID"
echo "App URL: https://${SPACE_ID/\//-}.hf.space"
echo "Watch build/runtime logs on Hugging Face. All app execution is on Hugging Face."

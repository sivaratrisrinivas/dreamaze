import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


_DEFAULT_SCRIPT_PATH = "jobs/dreamaze_hf_job.py"
_DEFAULT_PACKAGE = (
    "dreamaze @ git+https://github.com/sivaratrisrinivas/dreamaze.git"
)
BEST_GPU_HARDWARE_FLAVOR = "a100-large"


_DEFAULTS = {
    "hardware_flavor": "cpu-basic",
    "timeout": "30m",
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
    "training_seed": 0,
    "evaluation_seed": 0,
    "retry_count": 0,
    "eval_split": "validation",
    "device": "cpu",
    "precision": "float32",
    "num_workers": 0,
    "output_repo_type": "dataset",
    "output_path_in_repo": "dreamaze-runs/latest",
    "env": {},
}


_BEST_GPU_PROFILE = {
    "hardware_flavor": BEST_GPU_HARDWARE_FLAVOR,
    "timeout": "8h",
    "dataset_preset": "first",
    "batch_size": 64,
    "sampling_steps": 64,
    "max_train_steps": 5000,
    "checkpoint_every_steps": 250,
    "device": "cuda",
    "precision": "float16",
    "num_workers": 4,
    "env": {"HF_XET_HIGH_PERFORMANCE": "1"},
}


@dataclass(frozen=True)
class HuggingFaceJobConfig:
    hardware_flavor: str = "cpu-basic"
    timeout: str = "30m"
    python: str = "3.12"
    repo: str | None = None
    namespace: str | None = None
    script_path: str = _DEFAULT_SCRIPT_PATH
    package: str = _DEFAULT_PACKAGE
    dataset_preset: str | None = "tiny"
    dataset_dir: str = "/tmp/dreamaze/dataset"
    checkpoint_dir: str = "/tmp/dreamaze/checkpoints"
    evaluation_output: str = "/tmp/dreamaze/evaluation.json"
    batch_size: int = 1
    sampling_steps: int = 2
    max_train_steps: int = 1
    checkpoint_every_steps: int = 1
    learning_rate: float = 0.01
    training_seed: int = 0
    evaluation_seed: int = 0
    retry_count: int = 0
    eval_split: str = "validation"
    device: str = "cpu"
    precision: str = "float32"
    num_workers: int = 0
    output_repo: str | None = None
    output_repo_type: str = "dataset"
    output_path_in_repo: str = "dreamaze-runs/latest"
    env: Mapping[str, str] = field(default_factory=dict)


def load_huggingface_job_config(path: str | Path) -> HuggingFaceJobConfig:
    payload = json.loads(Path(path).read_text())
    payload = _resolve_compute_profile(payload)
    config = HuggingFaceJobConfig(
        hardware_flavor=payload.get("hardware_flavor", _DEFAULTS["hardware_flavor"]),
        timeout=payload.get("timeout", _DEFAULTS["timeout"]),
        python=payload.get("python", _DEFAULTS["python"]),
        repo=payload.get("repo"),
        namespace=payload.get("namespace"),
        script_path=payload.get("script_path", _DEFAULT_SCRIPT_PATH),
        package=payload.get("package", _DEFAULT_PACKAGE),
        dataset_preset=payload.get("dataset_preset", _DEFAULTS["dataset_preset"]),
        dataset_dir=payload.get("dataset_dir", _DEFAULTS["dataset_dir"]),
        checkpoint_dir=payload.get("checkpoint_dir", _DEFAULTS["checkpoint_dir"]),
        evaluation_output=payload.get(
            "evaluation_output", _DEFAULTS["evaluation_output"]
        ),
        batch_size=payload.get("batch_size", _DEFAULTS["batch_size"]),
        sampling_steps=payload.get("sampling_steps", _DEFAULTS["sampling_steps"]),
        max_train_steps=payload.get("max_train_steps", _DEFAULTS["max_train_steps"]),
        checkpoint_every_steps=payload.get(
            "checkpoint_every_steps", _DEFAULTS["checkpoint_every_steps"]
        ),
        learning_rate=payload.get("learning_rate", _DEFAULTS["learning_rate"]),
        training_seed=payload.get("training_seed", _DEFAULTS["training_seed"]),
        evaluation_seed=payload.get("evaluation_seed", _DEFAULTS["evaluation_seed"]),
        retry_count=payload.get("retry_count", _DEFAULTS["retry_count"]),
        eval_split=payload.get("eval_split", _DEFAULTS["eval_split"]),
        device=payload.get("device", _DEFAULTS["device"]),
        precision=payload.get("precision", _DEFAULTS["precision"]),
        num_workers=payload.get("num_workers", _DEFAULTS["num_workers"]),
        output_repo=payload.get("output_repo"),
        output_repo_type=payload.get(
            "output_repo_type", _DEFAULTS["output_repo_type"]
        ),
        output_path_in_repo=payload.get(
            "output_path_in_repo", _DEFAULTS["output_path_in_repo"]
        ),
        env=payload.get("env", _DEFAULTS["env"]),
    )
    _validate_huggingface_job_config(config)
    return config


def _resolve_compute_profile(payload: Mapping[str, object]) -> dict[str, object]:
    profile = payload.get("compute_profile")
    if profile is None:
        return dict(payload)
    if profile != "best_gpu":
        raise ValueError("Hugging Face job compute profile is invalid")

    resolved: dict[str, object] = dict(_BEST_GPU_PROFILE)
    resolved.update(payload)
    profile_env = dict(_BEST_GPU_PROFILE["env"])
    profile_env.update(payload.get("env", {}))
    resolved["env"] = profile_env
    return resolved


def build_huggingface_job_command(config: HuggingFaceJobConfig) -> list[str]:
    _validate_huggingface_job_config(config)
    command = [
        "hf",
        "jobs",
        "uv",
        "run",
        "--flavor",
        config.hardware_flavor,
        "--timeout",
        config.timeout,
        "--python",
        config.python,
        "--with",
        config.package,
        "--with",
        "huggingface-hub>=0.36",
        "--secrets",
        "HF_TOKEN",
    ]
    if config.repo is not None:
        command.extend(["--repo", config.repo])
    if config.namespace is not None:
        command.extend(["--namespace", config.namespace])
    for name, value in sorted(config.env.items()):
        command.extend(["--env", f"{name}={value}"])

    command.append(config.script_path)
    command.append("--")
    command.extend(_remote_script_args(config))
    return command


def _remote_script_args(config: HuggingFaceJobConfig) -> list[str]:
    args = [
        "--dataset-dir",
        config.dataset_dir,
        "--checkpoint-dir",
        config.checkpoint_dir,
        "--evaluation-output",
        config.evaluation_output,
        "--batch-size",
        str(config.batch_size),
        "--sampling-steps",
        str(config.sampling_steps),
        "--max-train-steps",
        str(config.max_train_steps),
        "--checkpoint-every-steps",
        str(config.checkpoint_every_steps),
        "--learning-rate",
        str(config.learning_rate),
        "--training-seed",
        str(config.training_seed),
        "--evaluation-seed",
        str(config.evaluation_seed),
        "--retry-count",
        str(config.retry_count),
        "--eval-split",
        config.eval_split,
        "--device",
        config.device,
        "--precision",
        config.precision,
        "--num-workers",
        str(config.num_workers),
    ]
    if config.dataset_preset is not None:
        args.extend(["--dataset-preset", config.dataset_preset])
    if config.output_repo is not None:
        args.extend(
            [
                "--output-repo",
                config.output_repo,
                "--output-repo-type",
                config.output_repo_type,
                "--output-path-in-repo",
                config.output_path_in_repo,
            ]
        )
    return args


def _validate_huggingface_job_config(config: HuggingFaceJobConfig) -> None:
    if config.dataset_preset not in {None, "tiny", "first"}:
        raise ValueError("Hugging Face job dataset preset must be tiny or first")
    if config.output_repo_type not in {"dataset", "model", "space"}:
        raise ValueError("Hugging Face job output repo type is invalid")
    if config.batch_size < 1:
        raise ValueError("Hugging Face job batch size must be positive")
    if config.sampling_steps < 1:
        raise ValueError("Hugging Face job sampling steps must be positive")
    if config.max_train_steps < 1:
        raise ValueError("Hugging Face job max train steps must be positive")
    if config.checkpoint_every_steps < 1:
        raise ValueError("Hugging Face job checkpoint cadence must be positive")
    if config.learning_rate <= 0:
        raise ValueError("Hugging Face job learning rate must be positive")
    if config.retry_count < 0:
        raise ValueError("Hugging Face job retry count cannot be negative")
    if config.num_workers < 0:
        raise ValueError("Hugging Face job worker count cannot be negative")
    if not config.hardware_flavor:
        raise ValueError("Hugging Face job hardware flavor is required")
    if not config.timeout:
        raise ValueError("Hugging Face job timeout is required")

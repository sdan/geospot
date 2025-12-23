"""
Modal job: Visual geolocation RL training with Tinker API.

CPU-only orchestrator - Tinker handles GPU remotely.
Streams OSV-5M directly from HuggingFace.

Usage:
    modal run modal_tinker.py             # Start training
    modal run modal_tinker.py --action train
    modal run modal_tinker.py --action show_cache
"""
import modal
import os

app = modal.App("geospot-tinker-vlm")

REPO_REMOTE_PATH = "/root/geospot-vlm"

# CPU-only image - Tinker handles GPU remotely
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "tinker",
        "datasets",  # For OSV-5M streaming
        "pillow",
        "wandb",
        "chz",
        "huggingface_hub",
        "numpy<2",
        "torch==2.2.2+cpu",
        "torchvision==0.17.2+cpu",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("geospot-vlm @ git+https://github.com/sdan/geospot-vlm.git")
    .add_local_dir(
        "geospot",
        remote_path=f"{REPO_REMOTE_PATH}/geospot",
        ignore=["__pycache__"],
    )
)

# HF cache volume for OSV-5M streaming cache
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

TRAIN_SECRETS = [
    modal.Secret.from_name("tinker-api-key"),
    modal.Secret.from_name("wandb-api-key"),
    modal.Secret.from_name("hf-token"),
]

# Training hyperparameters
COMMON_ARGS = {
    "max_steps": 100,
    "batch_size": 64,
    "group_size": 8,
    "learning_rate": 4e-5,
    "save_every": 25,
    "behavior_if_log_dir_exists": "delete",
}

SMOKE_ARGS = {
    "max_steps": 2,
    "batch_size": 2,
    "group_size": 2,
    "hf_repo": "osv5m/osv5m",
    "wandb_project": "None",
    "save_every": 0,
    "behavior_if_log_dir_exists": "delete",
}


def _run_with_repo(cmd: list[str]):
    """Run a command with the local repo path first on PYTHONPATH."""
    import subprocess

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_REMOTE_PATH}:{existing}" if existing else REPO_REMOTE_PATH
    subprocess.run(cmd, check=True, env=env)


def _build_train_cmd(extra_args: dict | None = None) -> list[str]:
    """Build training command using cookbook entry point."""
    cmd = [
        "python", "-m", "geospot.cookbook.train",
        "model_name=Qwen/Qwen3-VL-30B-A3B-Instruct",
        "hf_repo=osv5m/osv5m",
        f"max_steps={COMMON_ARGS['max_steps']}",
        f"batch_size={COMMON_ARGS['batch_size']}",
        f"group_size={COMMON_ARGS['group_size']}",
        f"learning_rate={COMMON_ARGS['learning_rate']}",
        f"save_every={COMMON_ARGS['save_every']}",
        "wandb_project=geospot-tinker",
        f"behavior_if_log_dir_exists={COMMON_ARGS['behavior_if_log_dir_exists']}",
    ]
    if extra_args:
        for k, v in extra_args.items():
            cmd.append(f"{k}={v}")
    return cmd


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,  # 24 hours
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    secrets=TRAIN_SECRETS,
)
def train(extra_args: dict | None = None):
    """Run visual geolocation RL training with OSV-5M."""
    cmd = _build_train_cmd(extra_args=extra_args)
    print(f"Running: {' '.join(cmd)}")
    _run_with_repo(cmd)
    hf_cache_volume.commit()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def show_hf_cache(max_entries: int = 50):
    """List HuggingFace cache contents in the persistent volume."""
    import pathlib

    cache_root = "/root/.cache/huggingface"
    if not os.path.exists(cache_root):
        print(f"No cache dir at {cache_root}")
        return

    files = []
    for path in pathlib.Path(cache_root).rglob("*"):
        if path.is_file():
            try:
                files.append((path.stat().st_size, str(path)))
            except OSError:
                continue

    files.sort(reverse=True)
    total_bytes = sum(size for size, _ in files)
    print(f"Total files: {len(files)}")
    print(f"Total size: {total_bytes / (1024 ** 3):.2f} GB")
    print(f"Top {min(max_entries, len(files))} files:")
    for size, path in files[:max_entries]:
        print(f"{size / (1024 ** 2):8.2f} MB  {path}")


@app.local_entrypoint()
def main(action: str = "train"):
    """
    Entry point for modal run.

    Usage:
        modal run modal_tinker.py                   # train (default)
        modal run modal_tinker.py --action train
        modal run modal_tinker.py --action show_cache
    """
    if action == "train":
        print("Starting visual geolocation RL training on OSV-5M...")
        train.remote()
        print("Training complete!")
    elif action == "smoke":
        print("Starting smoke run (2 steps) on OSV-5M...")
        train.remote(extra_args=SMOKE_ARGS)
        print("Smoke run complete!")
    elif action == "show_cache":
        show_hf_cache.remote()
    else:
        print(f"Unknown action: {action}")
        print("Options: train, show_cache")

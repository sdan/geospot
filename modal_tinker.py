"""
Modal jobs for geospot training with Tinker API.

CPU-only orchestrator - Tinker handles GPU remotely.
Streams OSV-5M directly from HuggingFace.

Usage:
    modal run modal_tinker.py                      # RL training (default)
    modal run modal_tinker.py --action rl
    modal run modal_tinker.py --action sft
    modal run modal_tinker.py --action smoke       # Quick test (2 steps)
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
        "datasets",
        "pillow",
        "wandb",
        "chz",
        "huggingface_hub",
        "transformers",
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

# =============================================================================
# Config presets
# =============================================================================

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
HF_REPO = "osv5m/osv5m"

# RL training defaults
RL_ARGS = {
    "model_name": MODEL,
    "hf_repo": HF_REPO,
    "max_steps": 100,
    "batch_size": 64,
    "group_size": 8,
    "learning_rate": 4e-5,
    "save_every": 25,
    "env_type": "single",  # "single" or "multi"
    "wandb_project": "geospot-tinker-dec23",
    "behavior_if_log_dir_exists": "delete",
}

# SFT training defaults
SFT_ARGS = {
    "model_name": MODEL,
    "hf_repo": HF_REPO,
    "max_steps": 1000,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "lora_rank": 32,
    "save_every": 100,
    "wandb_project": "geospot-tinker-dec23",
    "behavior_if_log_dir_exists": "delete",
}

# Quick smoke tests
SMOKE_RL_ARGS = {
    "max_steps": 2,
    "batch_size": 2,
    "group_size": 2,
    "save_every": 0,
    "wandb_project": "",
    "behavior_if_log_dir_exists": "delete",
}

SMOKE_SFT_ARGS = {
    "max_steps": 2,
    "batch_size": 2,
    "save_every": 0,
    "wandb_project": "",
    "behavior_if_log_dir_exists": "delete",
}


# =============================================================================
# Helpers
# =============================================================================


def _run_with_repo(cmd: list[str]):
    """Run a command with the local repo path first on PYTHONPATH."""
    import subprocess

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_REMOTE_PATH}:{existing}" if existing else REPO_REMOTE_PATH
    subprocess.run(cmd, check=True, env=env)


def _build_cmd(module: str, args: dict, extra_args: dict | None = None) -> list[str]:
    """Build training command for a module."""
    merged = {**args}
    if extra_args:
        merged.update(extra_args)

    cmd = ["python", "-m", module]
    for k, v in merged.items():
        if v is not None and v != "":
            cmd.append(f"{k}={v}")
    return cmd


# =============================================================================
# Modal functions
# =============================================================================


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,  # 24 hours
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=60.0),
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    secrets=TRAIN_SECRETS,
)
def train_rl(extra_args: dict | None = None):
    """Run GRPO RL training on OSV-5M."""
    cmd = _build_cmd("geospot.train_rl", RL_ARGS, extra_args)
    print(f"Running RL: {' '.join(cmd)}")
    _run_with_repo(cmd)
    hf_cache_volume.commit()


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,  # 24 hours
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=60.0),
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    secrets=TRAIN_SECRETS,
)
def train_sft(extra_args: dict | None = None):
    """Run SFT warm-start on OSV-5M."""
    cmd = _build_cmd("geospot.sft", SFT_ARGS, extra_args)
    print(f"Running SFT: {' '.join(cmd)}")
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
def main(action: str = "rl"):
    """
    Entry point for modal run.

    Usage:
        modal run modal_tinker.py --action all          # Run SFT + RL-single + RL-multi in parallel
        modal run modal_tinker.py --action rl-single    # GRPO RL single-turn
        modal run modal_tinker.py --action rl-multi     # GRPO RL multi-turn (dense rewards)
        modal run modal_tinker.py --action sft          # SFT warm-start
        modal run modal_tinker.py --action smoke-rl     # Quick RL test (2 steps)
        modal run modal_tinker.py --action smoke-sft    # Quick SFT test (2 steps)
        modal run modal_tinker.py --action show_cache
    """
    if action == "all":
        print("=" * 60)
        print("Starting ALL training runs in parallel (single Modal client)")
        print("=" * 60)
        print(f"\nSFT Config: {SFT_ARGS}")
        print(f"\nRL Single Config: {RL_ARGS}")
        print(f"\nRL Multi Config: {{**RL_ARGS, 'env_type': 'multi'}}")
        print("\nLaunching all 3 jobs...")

        # Spawn all 3 in parallel using .spawn() for non-blocking
        sft_handle = train_sft.spawn()
        rl_single_handle = train_rl.spawn(extra_args={"env_type": "single"})
        rl_multi_handle = train_rl.spawn(extra_args={"env_type": "multi"})

        print("\nAll jobs spawned! Waiting for completion...")
        print("Check wandb: https://wandb.ai/sdan/geospot-tinker-dec23\n")

        # Wait for all to complete
        sft_handle.get()
        print("✓ SFT complete!")
        rl_single_handle.get()
        print("✓ RL single-turn complete!")
        rl_multi_handle.get()
        print("✓ RL multi-turn complete!")

        print("\n" + "=" * 60)
        print("ALL TRAINING COMPLETE!")
        print("=" * 60)

    elif action == "rl" or action == "rl-single":
        print("Starting GRPO RL (single-turn) on OSV-5M...")
        print(f"Config: {RL_ARGS}")
        train_rl.remote(extra_args={"env_type": "single"})
        print("RL single-turn training complete!")
    elif action == "rl-multi":
        print("Starting GRPO RL (multi-turn dense rewards) on OSV-5M...")
        config = {**RL_ARGS, "env_type": "multi"}
        print(f"Config: {config}")
        train_rl.remote(extra_args={"env_type": "multi"})
        print("RL multi-turn training complete!")
    elif action == "sft":
        print("Starting SFT warm-start on OSV-5M...")
        print(f"Config: {SFT_ARGS}")
        train_sft.remote()
        print("SFT training complete!")
    elif action == "smoke-rl":
        print("Starting RL smoke test (2 steps)...")
        train_rl.remote(extra_args=SMOKE_RL_ARGS)
        print("RL smoke test complete!")
    elif action == "smoke-sft":
        print("Starting SFT smoke test (2 steps)...")
        train_sft.remote(extra_args=SMOKE_SFT_ARGS)
        print("SFT smoke test complete!")
    elif action == "show_cache":
        show_hf_cache.remote()
    else:
        print(f"Unknown action: {action}")
        print("Options: all, rl-single, rl-multi, sft, smoke-rl, smoke-sft, show_cache")

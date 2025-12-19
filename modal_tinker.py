"""
Modal job: Run geospot-vlm Tinker training using cached geomix data.

CPU-only - Tinker API handles all GPU work remotely.
First run probes for existing geomix cache locations.

Usage:
    modal run modal_tinker.py::probe_cache  # Find where geomix is cached
    modal run modal_tinker.py::train        # Run training
"""
import modal
import os

app = modal.App("geospot-tinker-vlm")

# CPU-only image - Tinker handles GPU remotely
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for pip install from GitHub
    .pip_install(
        "tinker",
        "webdataset",
        "pillow",
        "wandb",
        "chz",
        "huggingface_hub",
        "torch",
        "torchvision",
    )
    .pip_install("geospot-vlm @ git+https://github.com/sdan/geospot-vlm.git")
)

# Check common Modal volume mount points
CACHE_PATHS = [
    "/root/.cache/huggingface",
    "/cache",
    "/data",
    "/vol",
    "/mnt",
    os.path.expanduser("~/.cache"),
]


@app.function(image=image, cpu=4, memory=16384, timeout=300)
def probe_cache():
    """Probe for existing geomix cache locations."""
    import subprocess
    import glob

    print("=== Probing for geomix cache ===")
    print(f"Working dir: {os.getcwd()}")
    print(f"Home: {os.path.expanduser('~')}")

    # Check environment
    print("\n=== Environment ===")
    for key in sorted(os.environ.keys()):
        if any(x in key.lower() for x in ['cache', 'hf', 'home', 'data', 'vol']):
            print(f"  {key}={os.environ[key]}")

    # Check common paths
    print("\n=== Checking paths ===")
    for path in CACHE_PATHS:
        if os.path.exists(path):
            print(f"\n✓ {path} exists")
            try:
                result = subprocess.run(
                    ["find", path, "-name", "*.tar", "-type", "f"],
                    capture_output=True, text=True, timeout=30
                )
                tar_files = [f for f in result.stdout.strip().split('\n') if f]
                if tar_files:
                    print(f"  Found {len(tar_files)} .tar files:")
                    for f in tar_files[:5]:
                        print(f"    {f}")
                    if len(tar_files) > 5:
                        print(f"    ... and {len(tar_files) - 5} more")
            except Exception as e:
                print(f"  Error scanning: {e}")
        else:
            print(f"✗ {path} does not exist")

    # Check for geomix specifically
    print("\n=== Looking for geomix ===")
    for pattern in ["**/geomix/**/*.tar", "**/sdan/**/*.tar", "**/*geomix*.tar"]:
        matches = glob.glob(f"/root/{pattern}", recursive=True)
        matches += glob.glob(f"/{pattern}", recursive=True)
        if matches:
            print(f"Found geomix data: {matches[:3]}")
            return {"found": True, "paths": matches}

    print("No geomix cache found - will need to set up a volume")
    return {"found": False, "paths": []}


# Modal volume for persistent geomix cache
geomix_volume = modal.Volume.from_name("geomix-cache", create_if_missing=True)


TRAIN_SECRETS = [
    modal.Secret.from_name("tinker-api-key"),
    modal.Secret.from_name("wandb-api-key"),
    modal.Secret.from_name("hf-token"),
]

COMMON_ARGS = {
    "max_steps": 1000,
    "batch_size": 64,
    "group_size": 8,
    "learning_rate": 4e-5,
    "save_every": 50,
}


def _ensure_cache(cache_path: str):
    """Ensure geomix is cached, download if needed."""
    import subprocess
    import glob

    tar_files = glob.glob(f"{cache_path}/**/*.tar", recursive=True)
    print(f"Found {len(tar_files)} cached shards in {cache_path}")

    if len(tar_files) < 1000:
        print("Downloading geomix...")
        subprocess.run([
            "huggingface-cli", "download", "sdan/geomix",
            "--repo-type", "dataset",
            "--local-dir", cache_path,
        ], check=True)
        geomix_volume.commit()
        tar_files = glob.glob(f"{cache_path}/**/*.tar", recursive=True)
        print(f"Downloaded {len(tar_files)} shards")

    return len(tar_files)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,  # 24 hours
    volumes={"/cache/geomix": geomix_volume},
    secrets=TRAIN_SECRETS,
)
def train_single_step():
    """Run single-step GeoEnv training."""
    import subprocess

    cache_path = "/cache/geomix"
    _ensure_cache(cache_path)

    cmd = [
        "python", "-m", "geospot.train",
        "model_name=Qwen/Qwen3-VL-30B-A3B-Instruct",
        f"local_path={cache_path}",
        f"max_steps={COMMON_ARGS['max_steps']}",
        f"batch_size={COMMON_ARGS['batch_size']}",
        f"group_size={COMMON_ARGS['group_size']}",
        f"learning_rate={COMMON_ARGS['learning_rate']}",
        f"save_every={COMMON_ARGS['save_every']}",
        "wandb_project=geospot-tinker",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,
    volumes={"/cache/geomix": geomix_volume},
    secrets=TRAIN_SECRETS,
)
def train_hierarchical():
    """Run hierarchical (country -> coords) training."""
    import subprocess

    cache_path = "/cache/geomix"
    _ensure_cache(cache_path)

    cmd = [
        "python", "-m", "geospot.train_hierarchical",
        "model_name=Qwen/Qwen3-VL-30B-A3B-Instruct",
        f"local_path={cache_path}",
        f"max_steps={COMMON_ARGS['max_steps']}",
        f"batch_size={COMMON_ARGS['batch_size']}",
        f"group_size={COMMON_ARGS['group_size']}",
        f"learning_rate={COMMON_ARGS['learning_rate']}",
        f"save_every={COMMON_ARGS['save_every']}",
        "wandb_project=geospot-tinker",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,
    volumes={"/cache/geomix": geomix_volume},
    secrets=TRAIN_SECRETS,
)
def train_curriculum():
    """Run curriculum (geohash 3-turn) training."""
    import subprocess

    cache_path = "/cache/geomix"
    _ensure_cache(cache_path)

    cmd = [
        "python", "-m", "geospot.train_curriculum",
        "model_name=Qwen/Qwen3-VL-30B-A3B-Instruct",
        f"local_path={cache_path}",
        f"max_steps={COMMON_ARGS['max_steps']}",
        f"batch_size={COMMON_ARGS['batch_size']}",
        f"group_size={COMMON_ARGS['group_size']}",
        f"learning_rate={COMMON_ARGS['learning_rate']}",
        f"save_every={COMMON_ARGS['save_every']}",
        "wandb_project=geospot-tinker",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 24,
    volumes={"/cache/geomix": geomix_volume},
    secrets=TRAIN_SECRETS,
)
def train_telescoping():
    """Run telescoping (potential-based) training."""
    import subprocess

    cache_path = "/cache/geomix"
    _ensure_cache(cache_path)

    cmd = [
        "python", "-m", "geospot.train_telescoping",
        "model_name=Qwen/Qwen3-VL-30B-A3B-Instruct",
        f"local_path={cache_path}",
        f"max_steps={COMMON_ARGS['max_steps']}",
        f"batch_size={COMMON_ARGS['batch_size']}",
        f"group_size={COMMON_ARGS['group_size']}",
        f"learning_rate={COMMON_ARGS['learning_rate']}",
        f"save_every={COMMON_ARGS['save_every']}",
        "wandb_project=geospot-tinker",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600 * 12,  # 12 hours for full download
    volumes={"/cache/geomix": geomix_volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def download_geomix():
    """Download full geomix dataset (~2TB) to Modal volume."""
    import subprocess
    import glob
    import os

    cache_path = "/cache/geomix"
    os.makedirs(cache_path, exist_ok=True)

    # Check existing
    tar_files = glob.glob(f"{cache_path}/**/*.tar", recursive=True)
    print(f"Existing shards: {len(tar_files)}")

    if len(tar_files) >= 14000:
        print("Dataset already cached!")
        return len(tar_files)

    print("Downloading full geomix dataset...")
    print("This will take a while (~2TB)...")

    # Use huggingface-cli for resumable download
    subprocess.run([
        "huggingface-cli", "download", "sdan/geomix",
        "--repo-type", "dataset",
        "--local-dir", cache_path,
    ], check=True)

    # Commit to volume
    geomix_volume.commit()

    tar_files = glob.glob(f"{cache_path}/**/*.tar", recursive=True)
    print(f"Download complete! {len(tar_files)} shards cached.")
    return len(tar_files)


@app.local_entrypoint()
def main(action: str = "download"):
    """
    Entry point for modal run.

    Usage:
        modal run modal_tinker.py  # downloads geomix (default)
        modal run modal_tinker.py --action download
        modal run modal_tinker.py --action probe
        modal run modal_tinker.py --action train_all
    """
    if action == "download":
        print("Starting geomix download to Modal volume...")
        result = download_geomix.remote()
        print(f"\nDone! {result} shards cached.")
    elif action == "probe":
        result = probe_cache.remote()
        print(f"\nResult: {result}")
    elif action == "train_all":
        print("Starting all 4 training runs in parallel...")
        handles = [
            train_single_step.spawn(),
            train_hierarchical.spawn(),
            train_curriculum.spawn(),
            train_telescoping.spawn(),
        ]
        for i, h in enumerate(handles):
            h.get()
            print(f"Training {i+1}/4 complete")
        print("All training complete!")
    elif action == "train_single":
        train_single_step.remote()
    elif action == "train_hier":
        train_hierarchical.remote()
    elif action == "train_curr":
        train_curriculum.remote()
    elif action == "train_tele":
        train_telescoping.remote()
    else:
        print(f"Unknown action: {action}")
        print("Options: download, probe, train_all, train_single, train_hier, train_curr, train_tele")

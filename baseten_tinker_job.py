"""
Baseten job: Run geospot-vlm Tinker training using cached geomix data.

Uses T4 (cheapest GPU) since Tinker handles actual model training remotely.
Reads from $BT_PROJECT_CACHE_DIR/geomix (pre-cached via precache-geomix job).

Usage:
    cd /Users/sdan/Developer/geospot-vlm
    truss train push baseten_tinker_job.py
    truss train logs geospot/tinker-vlm
"""
from truss_train import definitions
from truss.base import truss_config

# Try H100-based project
project_name = "geospot/geomix-cache-v2"

BASE_IMAGE = "python:3.11-slim"

# Training command - reads from Baseten cache
TRAIN_CMD = "python -m geospot.train model_name=Qwen/Qwen3-VL-30B-A3B-Instruct local_path=/root/.cache/user_artifacts/geomix/train max_steps=1000 batch_size=64 group_size=8 learning_rate=4e-5 save_every=50 wandb_project=geospot-tinker"

training_runtime = definitions.Runtime(
    start_commands=[
        "echo '=== Tinker VLM Training ===' && pwd && ls -la",
        # Check cache
        "echo '=== Cache ===' && find $BT_PROJECT_CACHE_DIR/geomix -name '*.tar' 2>/dev/null | wc -l",
        # Install deps - pyproject.toml is at / (root)
        "pip install -q tinker webdataset pillow wandb chz huggingface_hub",
        "pip install -q -e /",
        # Run training with cached data
        TRAIN_CMD,
    ],
    environment_variables={
        "TINKER_API_KEY": definitions.SecretReference(name="TINKER_API_KEY"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=definitions.CacheConfig(enabled=True),
)

# H100 - cheaper than H200
training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name=project_name,
    job=training_job,
)

"""
Test job: Try running Tinker client without GPU accelerator.
Uses a different project name to not disturb running jobs.
"""
from truss_train import definitions
from truss.base import truss_config

# New project for CPU-only test
project_name = "geospot/tinker-cpu-test"

BASE_IMAGE = "python:3.11-slim"

TRAIN_CMD = "python -m geospot.train model_name=Qwen/Qwen3-VL-30B-A3B-Instruct local_path=/root/.cache/user_artifacts/geomix/train max_steps=10 batch_size=64 group_size=8 learning_rate=4e-5 wandb_project=geospot-tinker"

training_runtime = definitions.Runtime(
    start_commands=[
        "echo '=== CPU-only Tinker Test ===' && pwd",
        "echo '=== Cache ===' && find $BT_PROJECT_CACHE_DIR/geomix -name '*.tar' 2>/dev/null | wc -l || echo 'No cache found'",
        "pip install -q tinker webdataset pillow wandb chz huggingface_hub",
        "pip install -q -e /",
        TRAIN_CMD,
    ],
    environment_variables={
        "TINKER_API_KEY": definitions.SecretReference(name="TINKER_API_KEY"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=definitions.CacheConfig(enabled=True),
)

# No accelerator - CPU only
training_compute = definitions.Compute(
    node_count=1,
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

"""
Baseten job: Run geospot-vlm Telescoping (3-turn: coarse -> refine -> final) training.

Uses H100 with cached geomix data.
Multi-turn with potential-based rewards: r_t = S(d_t) - S(d_{t-1}), so sum = final score.

Usage:
    cd /Users/sdan/Developer/geospot-vlm
    truss train push baseten_telescoping_job.py
    truss train logs geospot/geomix-cache-v2
"""
from truss_train import definitions
from truss.base import truss_config

project_name = "geospot/geomix-cache-v2"

BASE_IMAGE = "python:3.11-slim"

# Training command - telescoping 3-turn with potential-based rewards
TRAIN_CMD = "python -m geospot.train_telescoping model_name=Qwen/Qwen3-VL-30B-A3B-Instruct local_path=/root/.cache/user_artifacts/geomix/train max_steps=1000 batch_size=64 group_size=8 learning_rate=4e-5 save_every=50 wandb_project=geospot-tinker"

training_runtime = definitions.Runtime(
    start_commands=[
        "echo '=== Telescoping VLM Training (coarse -> refine -> final) ===' && pwd",
        "echo '=== Cache ===' && find $BT_PROJECT_CACHE_DIR/geomix -name '*.tar' 2>/dev/null | wc -l",
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

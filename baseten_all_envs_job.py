"""
Baseten job: Run all 4 env types on single H100.

Runs sequentially: single-step -> hierarchical -> curriculum -> telescoping
Each gets 1000 steps, all log to geospot-tinker wandb project.
"""
from truss_train import definitions
from truss.base import truss_config

project_name = "geospot/geomix-cache-v2"

BASE_IMAGE = "python:3.11-slim"

LOCAL_PATH = "/root/.cache/user_artifacts/geomix/train"
COMMON_ARGS = f"local_path={LOCAL_PATH} max_steps=1000 batch_size=64 group_size=8 learning_rate=4e-5 save_every=50 wandb_project=geospot-tinker"

training_runtime = definitions.Runtime(
    start_commands=[
        "echo '=== All 4 Envs on Single H100 ==='",
        f"echo '=== Cache ===' && find $BT_PROJECT_CACHE_DIR/geomix -name '*.tar' 2>/dev/null | wc -l",
        "pip install -q tinker webdataset pillow wandb chz huggingface_hub",
        "pip install -q -e /",
        # Run all 4 in parallel with & and wait
        f"python -m geospot.train model_name=Qwen/Qwen3-VL-30B-A3B-Instruct {COMMON_ARGS} &",
        f"python -m geospot.train_hierarchical model_name=Qwen/Qwen3-VL-30B-A3B-Instruct {COMMON_ARGS} &",
        f"python -m geospot.train_curriculum model_name=Qwen/Qwen3-VL-30B-A3B-Instruct {COMMON_ARGS} &",
        f"python -m geospot.train_telescoping model_name=Qwen/Qwen3-VL-30B-A3B-Instruct {COMMON_ARGS} &",
        "wait",
        "echo '=== All training complete ==='",
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

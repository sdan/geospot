"""
Baseten job: Pre-cache geomix dataset on CPU-only project.

Downloads all geomix shards to $BT_PROJECT_CACHE_DIR/geomix for future runs.
No GPU needed - just downloads data.

Usage:
    cd /Users/sdan/Developer/geospot-vlm
    truss train push baseten_cpu_precache_job.py
    truss train logs geospot/tinker-cpu
"""
from truss_train import definitions

project_name = "geospot/tinker-cpu"

BASE_IMAGE = "python:3.11-slim"

# Download geomix shards to cache
CACHE_CMD = """
python -c "
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ.get('BT_PROJECT_CACHE_DIR', '/tmp/cache')
geomix_dir = os.path.join(cache_dir, 'geomix')
os.makedirs(geomix_dir, exist_ok=True)

print(f'Downloading geomix to {geomix_dir}...')
snapshot_download(
    repo_id='sdan/geomix',
    repo_type='dataset',
    local_dir=geomix_dir,
    local_dir_use_symlinks=False,
)
print('Done!')

# Count shards
import glob
shards = glob.glob(f'{geomix_dir}/**/*.tar', recursive=True)
print(f'Cached {len(shards)} shards')
"
"""

training_runtime = definitions.Runtime(
    start_commands=[
        "echo '=== Pre-caching geomix on CPU project ==='",
        "pip install -q huggingface_hub hf_xet",
        CACHE_CMD,
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=definitions.CacheConfig(enabled=True),
)

# CPU only - no accelerator
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

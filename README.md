# geospot-vlm

Train VLMs to predict geographic coordinates from images.

## What it does

Given a street-level image, the model outputs:
```
City: San Francisco
Region: California
Country: United States
Latitude: 37.7749
Longitude: -122.4194
```

Reward = exp(-distance_km / 25). Closer predictions get higher reward.

## Quick start

```bash
cd geospot-vlm
uv sync
export TINKER_API_KEY=sk-...

# SFT warm-start (optional but recommended)
uv run geospot-sft \
    hf_repo=osv5m/osv5m \
    model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    max_samples=1000 \
    log_path=./runs/sft

# RL training
uv run geospot-train \
    hf_repo=osv5m/osv5m \
    model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    log_path=./runs/rl
```

## Structure

```
geospot/
├── rl/
│   ├── types.py        # Env, EnvGroupBuilder, RLDataset
│   ├── geo_env.py      # GeoEnv: image -> prediction -> reward
│   ├── geo_reward.py   # haversine_km, parse_geo_response
│   └── geo_dataset.py  # GeoDatasetBuilder for HuggingFace
├── renderers.py        # Qwen3VLRenderer
├── completers.py       # TinkerTokenCompleter
├── sft.py              # SFT warm-start
└── train.py            # RL training
```

## Key concepts

**GeoEnv**: Single-turn environment. Shows image, expects location prediction.

**Reward**: `exp(-distance_km / tau)` where default tau=25km.
- <1km: 96% reward
- 25km: 37% reward
- 100km: 2% reward

**GRPO**: Multiple rollouts per image (group_size=4), center rewards across group.

## Config

```bash
# SFT
uv run geospot-sft \
    hf_repo=osv5m/osv5m \
    model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    batch_size=8 \
    learning_rate=5e-4 \
    num_epochs=1 \
    max_samples=10000

# RL
uv run geospot-train \
    hf_repo=osv5m/osv5m \
    model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    batch_size=16 \
    group_size=4 \
    learning_rate=5e-4 \
    lora_rank=32 \
    coord_tau=25.0
```

## Data format

HuggingFace dataset with columns:
- `image`: PIL Image
- `latitude` / `lat`: float
- `longitude` / `lon`: float
- `city`, `region`, `country`: str (optional)

## License

Apache 2.0

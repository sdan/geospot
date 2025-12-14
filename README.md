# geospot-vlm

Fine-tuning VLMs to predict locations from street view images.

## about

Given a street view image, predict the latitude and longitude. We use GRPO (Group Relative Policy Optimization) with a geodesic reward function that starts coarse and tightens over training.

The key idea: a model can't learn city-level precision from scratch—the gradient signal is too sparse when predictions are thousands of kilometers off. But if you start with a large distance scale (τ=2000km) and gradually decrease it (→25km), the model learns continent → country → region → city.

```
reward = exp(-distance_km / τ)
```

## approach

1. **SFT warm-start** — teach the model to output coordinates in a consistent format
2. **GRPO** — 16 rollouts per image, advantage centering, importance-weighted updates

Training runs on [tinker](https://tinker.dev) with parallel rollouts (2048 concurrent samples per step) and streams data from HuggingFace via WebDataset.

## install

```bash
uv sync
export TINKER_API_KEY=...
```

## run

```bash
# sft
uv run python -m geospot.sft model_name=Qwen/Qwen2.5-VL-3B-Instruct max_steps=1000

# grpo
uv run python -m geospot.train max_steps=100
```

## data

[sdan/geospot-vista9](https://huggingface.co/datasets/sdan/geospot-vista9) — 9M street view images with coordinates.

## references

- [Rainbolt](https://www.youtube.com/@georainbolt) — human-level GeoGuessr
- [GRPO](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization

# geospot-vlm

Teaching VLMs to play GeoGuessr.

## why

GeoGuessr drops you in a random street view. You look around. You guess where you are.

Humans get scary good at this. They read Cyrillic vs Latin scripts, spot Japanese kei trucks, notice Australian road signs have a specific font, know red soil + baobab = probably Africa. It's a weirdly rich visual reasoning task.

Can a VLM learn the same thing? Turns out: yes, if you shape the rewards right.

## the trick

Naive approach: reward = how close was the guess? Problem: model gets no gradient signal when it's 5000km off. It just flails.

What works: **progressive geodesic tightening**.

```
reward = exp(-distance_km / τ)
```

The insight: you can't learn city-level accuracy from scratch. But you can learn continent → country → region → city by starting with large τ and shrinking it.

- τ = 2000km — model learns hemispheres, continents
- τ = 500km — model learns countries, major regions
- τ = 100km — model learns states, provinces
- τ = 25km — model learns cities

Start coarse, end precise. The model picks up on signs, architecture, vegetation, road markings, sun position—same cues humans use.

## two-stage pipeline

**Stage 1: SFT warm-start.** Teach the model the output format. Without this, GRPO wastes steps learning to output valid coordinates instead of learning geography.

**Stage 2: GRPO.** Now the model knows *how* to answer. RL teaches it to answer *correctly*. Group sampling (16 rollouts per image), advantage centering, importance-weighted gradients.

Format compliance first, then precision.

## how it works

Built on [tinker](https://tinker.dev) for distributed LoRA training.

**Parallel rollouts.** Fire all sample requests before collecting any results. For batch=128, group=16, that's 2048 concurrent inference requests. ~2 min/step.

```python
# Fire ALL requests (non-blocking)
for ob, env in env_obs:
    for _ in range(group_size):
        future = sampling_client.sample(prompt=ob, ...)
        all_futures.append(future)

# Collect after
for future in all_futures:
    result = future.result()
```

**Streaming WebDataset.** 9M images streamed from HuggingFace. Shard shuffle + sample shuffle. Handles multi-heading panoramas (4 views per location).

**Haversine distance.** Great-circle on a sphere, not Euclidean. Matters at global scale.

## install

```bash
uv sync
export TINKER_API_KEY=...
```

## run

```bash
# stage 1: sft
uv run python -m geospot.sft model_name=Qwen/Qwen2.5-VL-3B-Instruct max_steps=1000

# stage 2: grpo
uv run python -m geospot.train max_steps=100
```

Key params:
- `batch_size=128` — environments per step
- `group_size=16` — rollouts per environment
- `coord_tau=25.0` — distance scale (km)

## data

`sdan/geospot-vista9` — 9M street view images with coordinates.

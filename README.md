# geospot-vlm

Teaching VLMs to play GeoGuessr.

## why

GeoGuessr drops you in a random street view. You look around. You guess where you are.

Humans get scary good at this. They learn to read Cyrillic vs Latin scripts, spot Japanese kei trucks, notice that Australian road signs have a specific font, know that red soil + baobab trees = probably Africa. It's a weirdly rich visual reasoning task.

Can a VLM learn the same thing? Turns out: yes, if you shape the rewards right.

## the trick

Naive approach: reward = how close was the guess? Problem: the model has no gradient signal when it's 5000km off. It just flails.

What works: **hierarchical geodesic tightening**. Start coarse, end precise.

```
reward = coord_weight * exp(-distance_km / τ) + hierarchy_weight * (country + region + city)
```

Early training: τ is large, hierarchy matters more. Model learns "this looks European." Later: τ shrinks, coordinates matter more. Model learns "this is specifically Barcelona."

## how it actually works

We use [tinker](https://tinker.dev) for distributed LoRA training. The architecture:

**Parallel rollouts.** The naive approach—sample one completion, compute reward, repeat—is painfully slow. Instead, we fire off all sample requests in parallel across the batch before collecting any results. For a batch of 128 environments with group size 16, that's 2048 concurrent inference requests. Tinker handles the scheduling.

```python
# Fire ALL requests (non-blocking)
for ob, env in env_obs:
    for _ in range(group_size):
        future = sampling_client.sample(prompt=ob, ...)
        all_futures.append(future)

# Collect results after
for future in all_futures:
    result = future.result()
```

**GRPO with importance sampling.** We use Group Relative Policy Optimization—sample multiple completions per prompt, center the rewards, use the advantage as weights. The loss function is importance-sampled so we can reuse the same completions across gradient steps.

**Streaming WebDataset.** 9 million images is too big to shuffle in memory. We stream shards from HuggingFace, shuffle within a buffer, decode on the fly. Handles network hiccups gracefully.

**Haversine reward.** Great-circle distance on a sphere, not Euclidean distance. Matters when you're comparing predictions across the globe.

## install

```bash
uv sync
export TINKER_API_KEY=...
```

## run

```bash
# sft warm-start (optional but helps convergence)
uv run python -m geospot.sft model_name=Qwen/Qwen2.5-VL-3B-Instruct max_steps=1000

# grpo
uv run python -m geospot.train max_steps=100
```

Key hyperparameters:
- `batch_size=128` — environments per step
- `group_size=16` — completions per environment (for GRPO)
- `coord_tau=25.0` — distance scale in km (smaller = sharper gradient)
- `coord_weight=0.7` — balance between coordinate distance and hierarchy matching

Output format:
```
City: San Francisco
Country: United States
Latitude: 37.7749
Longitude: -122.4194
```

## data

`sdan/geospot-vista9` — street view images with coordinates. WebDataset format, streamed from HuggingFace.

# geospot-vlm

Fine-tune VLMs for geolocation prediction using GRPO.

## About

To solve AGI, we must first solve GeoGuessr.

GeoGuessr drops you in a random street view and you guess where you are. Humans learn to read signs, architecture, road markings, vegetation, sun position. Can a VLM learn the same?

We found VLMs can learn geolocation through **progressive geodesic tightening** — shape rewards to learn country → region → city, blended with coordinate distance on a schedule. The model picks up on geographic cues and gets surprisingly good.

Built on top of tinker for distributed LoRA training. SFT warm-start, then GRPO.

## Installation

```bash
uv sync
export TINKER_API_KEY=...
```

## Usage

```bash
# SFT warm-start
uv run python -m geospot.sft model_name=Qwen/Qwen3-VL-30B-A3B-Instruct max_steps=1000

# GRPO
uv run python -m geospot.train load_checkpoint_path=tinker://<id>/weights/final max_steps=100
```

Data: `sdan/geospot-vista9` (default)

## Output Format

```
City: San Francisco
Country: United States
Latitude: 37.7749
Longitude: -122.4194
```

## Reward

Hierarchical reward combining coordinate distance with geographic hierarchy:

```
reward = w_coord * exp(-distance_km / τ) + w_country * country_match + w_region * region_match + w_city * city_match
```

Default: τ=25km, coord_weight=0.7, hierarchy_weight=0.3

# geospot-vlm

Fine-tuning VLMs to predict locations from street view images.

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/0b67e03d-b17a-453f-9b38-88a860aca140" width="100%">
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/72482367-8046-4554-8cb2-4f8f264ef08d" width="100%">
    </td>
  </tr>
</table>

## About

There are 24,901 miles on this earth, of which I'm unable to experience all walks of life in the literal sense. But I can pursue my dreams via Google Earth. Ever since I was a kid I've been fascinated by what you can find on there, which eventually took me to GeoGuessr. I'm not the best geo-guesser but it's a challenge meant to be benchmark-maxxed.

Over the past year I've made an iOS app using a cheap GeoCLIP embedding model trained contrastively between Image<>GPS. Since then I've trained a [~1B param geoguessing model](https://huggingface.co/sdan/geospot-base) and had my fair share of fun [reimplementing Qwen3 from scratch in JAX](https://github.com/sdan/Qwen3-VL-JAX). Tinker dropped vision support on Friday so I decided to spin up a proper GRPO environment for geolocation over the weekend. Also built a dashboard to visualize rollouts in realtime since I wanted to watch the model's guesses cluster and tighten over training.

## Approach

The key idea: a model can't learn city-level precision from scratch. The gradient signal is too sparse when predictions are thousands of kilometers off.

### Single-Turn (τ-scheduling)

Start with a large distance scale (τ=2000km) and gradually decrease it (→25km), so the model learns continent → country → region → city.

```
reward = exp(-distance_km / τ)
```

### Multi-Turn Geohash Curriculum (NEW)

A 3-turn approach using geohash precision levels. Each turn asks for progressively more precise coordinates:

| Turn | Precision | Geohash Chars | Approx. Accuracy |
|------|-----------|---------------|------------------|
| 1 | Coarse | 2 | ~600km |
| 2 | Medium | 4 | ~40km |
| 3 | Fine | 6 | ~1km |

**How it works:**
- Turn 1: "Where is this? Give an approximate estimate." → Model predicts rough coords
- Turn 2: "Your initial guess was (X, Y). Refine your estimate." → Model refines
- Turn 3: "Your refined guess was (X, Y). Give final coordinates." → Final prediction

**Reward:** Geohash prefix matching + improvement bonus when predictions get closer.

**Teacher Forcing:** Configurable probability (0.0-1.0) to show ground truth hints vs model's own predictions between turns. TF=0.5 gives a balanced mix.

```bash
# Run multi-turn curriculum training
python -m geospot.train_geohash_curriculum max_samples=5000
```

**Why geohash?** No need for country/region text labels. Purely spatial, mathematically precise. Works with any lat/lon dataset.

### Training Pipeline

1. **SFT warm-start**: teach the model to output coordinates in a consistent format
2. **GRPO**: 8 rollouts per image, advantage centering, importance-weighted updates

Training runs on [tinker](https://tinker.dev) with concurrent samples per step, streaming data from HuggingFace.

## Install

```bash
uv sync
export TINKER_API_KEY=...
```

## Run

```bash
# sft
uv run python -m geospot.sft model_name=Qwen/Qwen2.5-VL-3B-Instruct max_steps=50
# grpo
uv run python -m geospot.train load_checkpoint_path=tinker://
```

## Data

[sdan/geospot-unified](https://huggingface.co/datasets/sdan/geospot-unified): unified geolocation dataset with multiple sources (cvcities, streetview, osv5m, mp16pro, msls).

## References

- [Rainbolt](https://www.youtube.com/@georainbolt): human-level GeoGuessr
- [GRPO](https://arxiv.org/abs/2402.03300): Group Relative Policy Optimization

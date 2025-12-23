# Geospot VLM: Visual Geolocation RL

Train a VLM to predict geographic coordinates from streetview images using GRPO.

## Quick Start

```bash
# Local training (streams from HuggingFace)
python -m geospot.cookbook.train

# With custom hyperparameters
python -m geospot.cookbook.train \
    max_steps=500 \
    batch_size=32 \
    group_size=4 \
    learning_rate=1e-4

# Resume from checkpoint
python -m geospot.cookbook.train \
    load_checkpoint_path="tinker://your-checkpoint-path"
```

## Modal Deployment

```bash
# Train on Modal (CPU orchestrator, Tinker handles GPU)
modal run modal_tinker.py --action train_all
```

## Architecture

**Single-turn RL environment:**
```
Image → VLM → "Latitude: 37.7749\nLongitude: -122.4194" → Reward
```

**Reward function:**
- **Distance reward**: `exp(-distance_km / tau)` with tau annealing
- **Geocell reward**: Geohash prefix match (hierarchical bonus)
- **Format penalty**: -0.1 for unparseable responses

**Tau schedule** (coarse → fine):
- Start: τ=2000 (continent-level signal)
- End: τ=25 (city-level precision)

## Expected Metrics

After 1000 steps on OSV-5M:
- `reward/mean`: 0.3 → 0.6
- `distance_km/mean`: 5000km → 500km
- Training time: ~4 hours

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Base model |
| `lora_rank` | 32 | LoRA rank |
| `hf_repo` | `osv5m/osv5m` | Dataset (OSV-5M or sdan/geomix) |
| `batch_size` | 64 | Images per step |
| `group_size` | 8 | Samples per image (GRPO) |
| `max_steps` | 1000 | Training steps |
| `learning_rate` | 4e-5 | Adam LR |
| `coord_tau_start` | 2000 | Initial τ (coarse) |
| `coord_tau_end` | 25 | Final τ (fine) |
| `coord_weight` | 0.7 | Distance reward weight |
| `geocell_weight` | 0.3 | Geohash reward weight |

## Sample Output

```
<think>
The architecture appears to be European style with the ornate window
frames and street layout. The signage visible suggests Portuguese...
</think>
Latitude: 38.7167
Longitude: -9.1395
```

Ground truth: Lisbon, Portugal (38.7223, -9.1393)
Distance: 0.6 km
Reward: 0.98

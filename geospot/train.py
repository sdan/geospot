"""
GRPO training for geospot VLM.

Single-step + geocell hierarchy + tau schedule (Schulman-style).
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime

import chz
import tinker
import torch
import wandb
from tinker import TensorData
from tinker.types import AdamParams

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.db import DBWriter
from geospot.rl.geo_reward import parse_geo_response
from geospot.renderers import get_renderer
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.geo_dataset import StreamingGeoDataset
from geospot.rl.geo_env import GeoEnv, GeoEnvConfig
from geospot.rl.geo_reward import GeoRewardConfig

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI config for geospot RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "osv5m/osv5m"
    max_shards: int | None = None
    max_steps: int = 100
    local_path: str | None = None  # Local cache path (e.g., Modal volume: /cache/osv5m)

    # Training
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    max_tokens: int = 256
    temperature: float = 1.0

    # Reward + tau schedule
    coord_tau_start: float = 2000.0  # Coarse at start (continent-level signal)
    coord_tau_end: float = 25.0  # Fine at end (city-level precision)
    use_tau_schedule: bool = True
    coord_reward_kind: str = "exp"  # "exp" or "geoguessr"
    coord_weight: float = 0.7
    geocell_weight: float = 0.3
    geohash_precision: int = 5

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-vlm"

    # Checkpointing
    save_every: int = 50
    load_checkpoint_path: str | None = None

    # Misc
    seed: int = 0
    base_url: str | None = None
    behavior_if_log_dir_exists: LogdirBehavior = "ask"


def compute_tau(step: int, max_steps: int, tau_start: float, tau_end: float) -> float:
    """Exponential tau schedule: coarse → fine."""
    if max_steps <= 1:
        return tau_end
    progress = step / (max_steps - 1)
    return tau_start * ((tau_end / tau_start) ** progress)


async def sample_group(
    sampling_client: tinker.SamplingClient,
    observation: tinker.ModelInput,
    group_size: int,
    sampling_params: tinker.SamplingParams,
) -> list[tuple[list[int], list[float]]]:
    """Sample group_size responses in single call. Returns [(tokens, logprobs), ...]."""
    result = await sampling_client.sample_async(
        prompt=observation,
        num_samples=group_size,
        sampling_params=sampling_params,
    )
    return [(seq.tokens, seq.logprobs) for seq in result.sequences]


async def run_training(cli: CLIConfig):
    """Main training loop."""
    # Setup logging
    if cli.log_path:
        log_path = cli.log_path
    else:
        model_name = cli.model_name.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_path = f"/tmp/geospot-rl/{model_name}-{timestamp}"

    check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"GRPO Training: {cli.hf_repo} -> {log_path}")
    logger.info(f"Model: {cli.model_name}, batch={cli.batch_size}, group={cli.group_size}")
    logger.info(f"Tau schedule: {cli.coord_tau_start} -> {cli.coord_tau_end} over {cli.max_steps} steps")

    if cli.wandb_project:
        wandb.init(
            project=cli.wandb_project,
            config={
                "model_name": cli.model_name,
                "batch_size": cli.batch_size,
                "group_size": cli.group_size,
                "learning_rate": cli.learning_rate,
                "coord_tau_start": cli.coord_tau_start,
                "coord_tau_end": cli.coord_tau_end,
                "use_tau_schedule": cli.use_tau_schedule,
                "coord_reward_kind": cli.coord_reward_kind,
                "coord_weight": cli.coord_weight,
                "geocell_weight": cli.geocell_weight,
                "geohash_precision": cli.geohash_precision,
                "max_tokens": cli.max_tokens,
            },
        )

    # Initialize components
    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    # Base reward config (tau will be updated each step)
    reward_config = GeoRewardConfig(
        coord_tau=cli.coord_tau_end,
        coord_reward_kind=cli.coord_reward_kind,
        coord_weight=cli.coord_weight,
        geocell_weight=cli.geocell_weight,
        geohash_precision=cli.geohash_precision,
    )
    env_config = GeoEnvConfig(reward_config=reward_config)

    dataset = StreamingGeoDataset(
        hf_repo=cli.hf_repo,
        group_size=cli.group_size,
        renderer=renderer,
        env_config=env_config,
        max_shards=cli.max_shards,
        seed=cli.seed,
        local_path=cli.local_path,
    )

    # Training client (use async variants to avoid deadlocks)
    service_client = tinker.ServiceClient(base_url=cli.base_url)
    if cli.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(cli.load_checkpoint_path)
        logger.info(f"Loaded checkpoint: {cli.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(cli.model_name, rank=cli.lora_rank)

    sampling_params = tinker.SamplingParams(
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = AdamParams(learning_rate=cli.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # Initialize viz DB writer
    run_id = str(uuid.uuid4())[:8]
    db = DBWriter(
        run_id=run_id,
        run_name=cli.hf_repo,
        run_type="rl",
        config={
            "model_name": cli.model_name,
            "hf_repo": cli.hf_repo,
            "lora_rank": cli.lora_rank,
            "batch_size": cli.batch_size,
            "group_size": cli.group_size,
            "learning_rate": cli.learning_rate,
            "coord_tau_start": cli.coord_tau_start,
            "coord_tau_end": cli.coord_tau_end,
            "max_steps": cli.max_steps,
        },
    )
    logger.info(f"Viz dashboard: http://localhost:3001/training-run/{run_id}")

    # Training loop
    for step in range(cli.max_steps):
        t_start = time.time()

        # Update tau (coarse → fine)
        if cli.use_tau_schedule:
            reward_config.coord_tau = compute_tau(step, cli.max_steps, cli.coord_tau_start, cli.coord_tau_end)
        else:
            reward_config.coord_tau = cli.coord_tau_end

        # Get fresh sampling client (combined async call is more efficient)
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"{step:06d}")

        # Get batch of env builders
        builders = dataset.get_batch(cli.batch_size)
        if not builders:
            logger.warning(f"Step {step}: no data")
            continue

        # Build environments and get observations
        envs: list[GeoEnv] = []
        observations: list[tinker.ModelInput] = []
        for builder in builders:
            env_group = await builder.make_envs()
            env = env_group[0]
            envs.append(env)
            ob, _ = await env.initial_observation()
            observations.append(ob)

        # Sample in parallel across all envs and groups
        all_samples: list[list[tuple[list[int], list[float]]]] = await asyncio.gather(
            *[sample_group(sampling_client, ob, cli.group_size, sampling_params) for ob in observations]
        )

        # Compute rewards and build training datums
        training_datums: list[tinker.Datum] = []
        batch_rewards: list[float] = []
        batch_distances: list[float] = []
        skipped_uniform = 0

        for group_idx, (env, ob, group_samples) in enumerate(zip(envs, observations, all_samples)):
            group_rewards: list[float] = []
            group_distances: list[float] = []
            sample_results: list[tuple[int, dict]] = []  # (sample_idx, metrics) for DB logging

            # Log image once per group
            image_id = db.log_image(
                step=step,
                group_idx=group_idx,
                image=env.image,
                gt_lat=env.ground_truth.lat,
                gt_lon=env.ground_truth.lon,
                gt_city=getattr(env.ground_truth, "city", None),
                gt_country=getattr(env.ground_truth, "country", None),
            )

            # Compute reward for each sample in group
            for sample_idx, (tokens, logprobs) in enumerate(group_samples):
                sample_env = GeoEnv(
                    image=env.image,
                    ground_truth=env.ground_truth,
                    renderer=env.renderer,
                    config=env.config,
                )
                step_result = await sample_env.step(tokens)
                group_rewards.append(step_result.reward)
                if "distance_km" in step_result.metrics:
                    group_distances.append(step_result.metrics["distance_km"])

                # Parse prediction for DB logging
                decoded_text = tokenizer.decode(tokens)
                parsed = parse_geo_response(decoded_text)
                db.log_sample(
                    image_id=image_id,
                    sample_idx=sample_idx,
                    pred_lat=parsed.location.lat if parsed.location else None,
                    pred_lon=parsed.location.lon if parsed.location else None,
                    pred_text=decoded_text[:500],  # Truncate for DB
                    distance_km=step_result.metrics.get("distance_km"),
                    reward=step_result.reward,
                    format_valid=parsed.format_valid,
                    mean_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
                )

            # Group-centered advantages (GRPO)
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]

            batch_rewards.append(mean_reward)
            if group_distances:
                batch_distances.append(sum(group_distances) / len(group_distances))

            # Skip if all rewards identical (no gradient signal)
            if all(a == 0.0 for a in advantages):
                skipped_uniform += 1
                continue

            # Build datums
            ob_len = ob.length - 1
            for (tokens, logprobs), advantage in zip(group_samples, advantages):
                full_chunks = list(ob.chunks) + [tinker.EncodedTextChunk(tokens=tokens[:-1])]
                full_input = tinker.ModelInput(chunks=full_chunks)
                full_targets = [0] * ob_len + tokens
                all_logprobs = [0.0] * ob_len + logprobs
                all_advantages = [0.0] * ob_len + [advantage] * len(logprobs)

                assert full_input.length == len(full_targets) == len(all_logprobs) == len(all_advantages)

                datum = tinker.Datum(
                    model_input=full_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(full_targets)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        if not training_datums:
            logger.warning(f"Step {step}: no training datums (all uniform rewards)")
            continue

        # Train
        fwd_bwd = training_client.forward_backward(training_datums, loss_fn="importance_sampling")
        optim = training_client.optim_step(adam_params)
        fwd_bwd.result()
        optim.result()

        # Metrics
        mean_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
        mean_dist = sum(batch_distances) / len(batch_distances) if batch_distances else 0
        elapsed = time.time() - t_start

        logger.info(
            f"Step {step}: reward={mean_reward:.3f}, dist={mean_dist:.0f}km, "
            f"tau={reward_config.coord_tau:.0f}, datums={len(training_datums)}, "
            f"skipped={skipped_uniform}, time={elapsed:.1f}s"
        )

        if cli.wandb_project:
            wandb.log(
                {
                    "progress/step": step,
                    "reward/mean": mean_reward,
                    "distance_km/mean": mean_dist,
                    "optim/coord_tau": reward_config.coord_tau,
                    "optim/datums": len(training_datums),
                    "optim/skipped_uniform": skipped_uniform,
                    "time/step_s": elapsed,
                }
            )

        # Log to viz DB
        db.log_step(
            step=step,
            mean_reward=mean_reward,
            mean_distance_km=mean_dist,
            coord_tau=reward_config.coord_tau,
            num_datums=len(training_datums),
            elapsed_s=elapsed,
        )

        # Checkpoint
        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved checkpoint: step_{step:06d}")

    # Final checkpoint
    result = training_client.save_state(name="final").result()
    db.close()
    logger.info(f"Training complete! Checkpoint: {result.path}")

    if cli.wandb_project:
        wandb.finish()


def main(cli: CLIConfig):
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_training(cli))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

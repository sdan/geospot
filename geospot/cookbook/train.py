"""
Visual geolocation RL training with Qwen3-VL.

Single-turn environment: image -> location prediction -> distance reward.
Uses OSV-5M dataset streaming and GRPO (group-relative policy optimization).
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Iterator, Literal

import chz
import tinker
import torch
import wandb
from tinker import TensorData
from tinker.types import AdamParams

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.data import iterate_samples, GeoSample
from geospot.renderers import get_renderer
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.geo_env import GeoEnv, GeoEnvConfig
from geospot.rl.geo_reward import GeoLocation, GeoRewardConfig, parse_geo_response

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for geolocation RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"
    load_checkpoint_path: str | None = None

    # Data
    hf_repo: str = "osv5m/osv5m"
    seed: int = 0
    max_shards: int | None = None
    local_path: str | None = None

    # Training
    batch_size: int = 64
    group_size: int = 8
    max_steps: int = 1000
    learning_rate: float = 4e-5
    max_tokens: int = 256
    temperature: float = 1.0

    # Reward (tau schedule: coarse -> fine over training)
    coord_tau_start: float = 2000.0  # Continent-level at start
    coord_tau_end: float = 25.0      # City-level at end
    coord_reward_kind: Literal["exp", "geoguessr"] = "exp"
    coord_weight: float = 0.7
    geocell_weight: float = 0.3
    geohash_precision: int = 5

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-vlm"
    wandb_name: str | None = None

    # Checkpointing
    save_every: int = 50
    eval_every: int = 0  # Not implemented yet

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: LogdirBehavior = "ask"


def compute_tau(step: int, max_steps: int, tau_start: float, tau_end: float) -> float:
    """Exponential tau schedule: coarse -> fine."""
    if max_steps <= 1:
        return tau_end
    progress = step / (max_steps - 1)
    return tau_start * ((tau_end / tau_start) ** progress)


def _align_sample(tokens: list[int], logprobs: list[float]) -> tuple[list[int], list[float]] | None:
    if not tokens or not logprobs:
        return None
    if len(tokens) != len(logprobs):
        seq_len = min(len(tokens), len(logprobs))
        if seq_len == 0:
            return None
        tokens = tokens[:seq_len]
        logprobs = logprobs[:seq_len]
    return tokens, logprobs


def _next_sample_with_retries(
    sample_iter: Iterator[GeoSample],
    cli: CLIConfig,
    step: int,
    max_attempts: int = 3,
) -> tuple[Iterator[GeoSample], GeoSample | None]:
    for attempt in range(max_attempts):
        try:
            return sample_iter, next(sample_iter)
        except StopIteration:
            sample_iter = iterate_samples(
                hf_repo=cli.hf_repo,
                seed=cli.seed + step + attempt + 1,
                shuffle_buffer=1000,
                max_shards=cli.max_shards,
                local_path=cli.local_path,
            )
    logger.warning("Failed to fetch a sample after %d attempts.", max_attempts)
    return sample_iter, None


async def sample_group(
    sampling_client: tinker.SamplingClient,
    observation: tinker.ModelInput,
    group_size: int,
    sampling_params: tinker.SamplingParams,
) -> list[tuple[list[int], list[float]]]:
    """Sample group_size responses. Returns [(tokens, logprobs), ...]."""
    result = await sampling_client.sample_async(
        prompt=observation,
        num_samples=group_size,
        sampling_params=sampling_params,
    )
    return [(seq.tokens, seq.logprobs) for seq in result.sequences]


async def cli_main(cli: CLIConfig):
    """Main training loop."""
    # Setup log path
    model_slug = cli.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"geo-{model_slug}-{cli.lora_rank}rank-{cli.learning_rate}lr-{timestamp}"

    if cli.log_path:
        log_path = cli.log_path
    else:
        log_path = f"/tmp/geospot-rl/{run_name}"

    check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"GRPO Training: {cli.hf_repo} -> {log_path}")
    logger.info(f"Model: {cli.model_name}, batch={cli.batch_size}, group={cli.group_size}")
    logger.info(f"Tau schedule: {cli.coord_tau_start} -> {cli.coord_tau_end}")

    # WandB
    if cli.wandb_project:
        wandb.init(
            project=cli.wandb_project,
            name=cli.wandb_name or run_name,
            config=chz.asdict(cli),
        )

    # Components
    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    # Reward config (tau updated per step)
    reward_config = GeoRewardConfig(
        coord_tau=cli.coord_tau_end,
        coord_reward_kind=cli.coord_reward_kind,
        coord_weight=cli.coord_weight,
        geocell_weight=cli.geocell_weight,
        geohash_precision=cli.geohash_precision,
    )
    env_config = GeoEnvConfig(reward_config=reward_config)

    # Data iterator
    sample_iter = iterate_samples(
        hf_repo=cli.hf_repo,
        seed=cli.seed,
        shuffle_buffer=1000,
        max_image_size=512,
        max_shards=cli.max_shards,
        local_path=cli.local_path,
    )

    # Training client
    service_client = tinker.ServiceClient(base_url=cli.base_url)
    if cli.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cli.load_checkpoint_path
        )
        logger.info(f"Resumed from: {cli.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cli.model_name, rank=cli.lora_rank
        )

    sampling_params = tinker.SamplingParams(
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = AdamParams(learning_rate=cli.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # Training loop
    for step in range(cli.max_steps):
        t_start = time.time()

        # Checkpoint
        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved: step_{step:06d}")

        # Update tau (coarse -> fine)
        reward_config.coord_tau = compute_tau(
            step, cli.max_steps, cli.coord_tau_start, cli.coord_tau_end
        )

        # Get sampling client from current weights
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"{step:06d}"
        )

        # Collect batch of samples
        batch_samples: list[GeoSample] = []
        for _ in range(cli.batch_size):
            sample_iter, sample = _next_sample_with_retries(sample_iter, cli, step)
            if sample is None:
                break
            batch_samples.append(sample)

        if not batch_samples:
            logger.error("No samples available; aborting training.")
            return

        # Build envs and observations
        envs: list[GeoEnv] = []
        observations: list[tinker.ModelInput] = []
        for sample in batch_samples:
            gt = GeoLocation(lat=sample.lat, lon=sample.lon, country=sample.country)
            env = GeoEnv(image=sample.image, ground_truth=gt, renderer=renderer, config=env_config)
            envs.append(env)
            ob, _ = await env.initial_observation()
            observations.append(ob)

        # Sample in parallel
        all_samples = await asyncio.gather(
            *[sample_group(sampling_client, ob, cli.group_size, sampling_params)
              for ob in observations]
        )

        # Compute rewards and build datums
        training_datums: list[tinker.Datum] = []
        batch_rewards: list[float] = []
        batch_distances: list[float] = []
        skipped_uniform = 0

        for env, ob, group_samples in zip(envs, observations, all_samples):
            group_rewards: list[float] = []
            group_distances: list[float] = []
            aligned_samples: list[tuple[list[int], list[float]]] = []

            for tokens, logprobs in group_samples:
                aligned = _align_sample(tokens, logprobs)
                if aligned is None:
                    continue
                tokens, logprobs = aligned
                sample_env = GeoEnv(
                    image=env.image,
                    ground_truth=env.ground_truth,
                    renderer=env.renderer,
                    config=env.config,
                )
                step_result = await sample_env.step(tokens)
                group_rewards.append(step_result.reward)
                aligned_samples.append((tokens, logprobs))
                if "distance_km" in step_result.metrics:
                    group_distances.append(step_result.metrics["distance_km"])

            # GRPO: group-centered advantages
            if not group_rewards:
                skipped_uniform += 1
                continue
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]

            batch_rewards.append(mean_reward)
            if group_distances:
                batch_distances.append(sum(group_distances) / len(group_distances))

            # Skip uniform rewards (no gradient signal)
            if all(a == 0.0 for a in advantages):
                skipped_uniform += 1
                continue

            # Build datums
            ob_len = max(0, ob.length - 1)
            for (tokens, logprobs), advantage in zip(aligned_samples, advantages):
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
            logger.warning(f"Step {step}: no datums (all uniform)")
            continue

        # Training step
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
            wandb.log({
                "progress/step": step,
                "reward/mean": mean_reward,
                "distance_km/mean": mean_dist,
                "optim/coord_tau": reward_config.coord_tau,
                "optim/datums": len(training_datums),
                "optim/skipped_uniform": skipped_uniform,
                "time/step_s": elapsed,
            })

    # Final checkpoint
    result = training_client.save_state(name="final").result()
    logger.info(f"Training complete! Checkpoint: {result.path}")

    if cli.wandb_project:
        wandb.finish()


def main(cli: CLIConfig):
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(cli_main(cli))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)

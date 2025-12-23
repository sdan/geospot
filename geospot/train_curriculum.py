"""
Multi-turn curriculum GRPO training for geospot VLM.

Supports:
- GeohashCurriculumEnv: coarse → medium → fine (geohash shaping)
- TelescopingGeoEnv: coarse → refine → final (telescoping distance shaping)
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
from geospot.data import iterate_samples, GeoSample
from geospot.renderers import get_renderer
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.types import Env
from geospot.rl.geohash_curriculum_env import GeohashCurriculumEnv, GeohashCurriculumConfig
from geospot.rl.telescoping_geo_env import TelescopingGeoEnv, TelescopingGeoEnvConfig
from geospot.rl.geo_reward import GeoLocation

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI config for curriculum RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "osv5m/osv5m"
    max_shards: int | None = None
    max_steps: int = 100
    local_path: str | None = None  # Local cache path (e.g., Modal volume: /cache/osv5m)

    # Training
    batch_size: int = 64  # Reduced for multi-turn (more tokens per episode)
    group_size: int = 8
    learning_rate: float = 4e-5
    max_tokens: int = 128  # Per turn (shorter for multi-turn)
    temperature: float = 1.0

    # Curriculum config
    env_kind: str = "geohash"  # "telescoping" or "geohash"
    improvement_bonus: float = 0.1
    telescoping_score_kind: str = "geoguessr"  # "geoguessr" or "exp"
    telescoping_exp_tau_km: float = 2000.0
    format_penalty: float = 0.1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-curriculum"

    # Checkpointing
    save_every: int = 25
    load_checkpoint_path: str | None = None

    # Misc
    seed: int = 0
    base_url: str | None = None
    behavior_if_log_dir_exists: LogdirBehavior = "ask"


async def sample_single(
    sampling_client: tinker.SamplingClient,
    observation: tinker.ModelInput,
    sampling_params: tinker.SamplingParams,
) -> tuple[list[int], list[float]]:
    """Sample a single response."""
    result = await sampling_client.sample_async(
        prompt=observation,
        num_samples=1,
        sampling_params=sampling_params,
    )
    seq = result.sequences[0]
    return seq.tokens, seq.logprobs


async def run_episode(
    env: Env,
    sampling_client: tinker.SamplingClient,
    sampling_params: tinker.SamplingParams,
) -> tuple[list[tuple[tinker.ModelInput, list[int], list[float], float]], dict]:
    """
    Run a full multi-turn episode.

    Returns:
        transitions: List of (observation, tokens, logprobs, reward) per turn
        final_metrics: Aggregated metrics from all turns
    """
    transitions = []
    final_metrics = {}

    # Initial observation
    obs, _stop_condition = await env.initial_observation()

    while True:
        # Sample response
        tokens, logprobs = await sample_single(sampling_client, obs, sampling_params)

        # Step environment
        step_result = await env.step(tokens)

        # Store transition
        transitions.append((obs, tokens, logprobs, step_result.reward))

        # Update metrics
        for k, v in step_result.metrics.items():
            if isinstance(v, (int, float)):
                final_metrics[k] = v

        if step_result.episode_done:
            break

        # Next observation
        obs = step_result.next_observation

    return transitions, final_metrics


async def run_training(cli: CLIConfig):
    """Main training loop for multi-turn curriculum."""
    # Setup logging
    if cli.log_path:
        log_path = cli.log_path
    else:
        model_name = cli.model_name.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_path = f"/tmp/geospot-curriculum/{model_name}-{timestamp}"

    check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"Curriculum GRPO Training: {cli.hf_repo} -> {log_path}")
    logger.info(f"Model: {cli.model_name}, batch={cli.batch_size}, group={cli.group_size}")
    logger.info(f"Env: {cli.env_kind}")
    if cli.env_kind == "geohash":
        logger.info("Multi-turn curriculum: coarse (600km) → medium (40km) → fine (1km)")
    elif cli.env_kind == "telescoping":
        logger.info(f"Telescoping shaping with score_kind={cli.telescoping_score_kind}")
    if cli.local_path:
        logger.info(f"Using local cache: {cli.local_path}")

    if cli.wandb_project:
        wandb.init(
            project=cli.wandb_project,
            name=f"curriculum-{cli.env_kind}",
            tags=["curriculum", cli.env_kind, "multi-turn"],
            config={
                "model_name": cli.model_name,
                "batch_size": cli.batch_size,
                "group_size": cli.group_size,
                "learning_rate": cli.learning_rate,
                "improvement_bonus": cli.improvement_bonus,
                "max_tokens": cli.max_tokens,
                "env_kind": cli.env_kind,
                "env_type": "curriculum",
            },
        )

    # Initialize components
    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    env_config = GeohashCurriculumConfig(
        improvement_bonus=cli.improvement_bonus,
    )
    telescoping_env_config = TelescopingGeoEnvConfig(
        score_kind=cli.telescoping_score_kind,
        exp_tau_km=cli.telescoping_exp_tau_km,
        format_penalty=cli.format_penalty,
    )

    # Streaming dataset
    sample_iter = iterate_samples(
        hf_repo=cli.hf_repo,
        max_shards=cli.max_shards,
        seed=cli.seed,
        local_path=cli.local_path,
    )

    # Training client
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
        run_name=f"{cli.hf_repo}-curriculum",
        run_type="rl-curriculum",
        config={
            "model_name": cli.model_name,
            "hf_repo": cli.hf_repo,
            "lora_rank": cli.lora_rank,
            "batch_size": cli.batch_size,
            "group_size": cli.group_size,
            "learning_rate": cli.learning_rate,
            "max_steps": cli.max_steps,
            "env_kind": cli.env_kind,
        },
    )
    logger.info(f"Viz dashboard: http://localhost:3001/training-run/{run_id}")

    # Helper to create env from sample
    def create_env_from_sample(sample: GeoSample) -> Env:
        ground_truth = GeoLocation(lat=sample.lat, lon=sample.lon)
        if cli.env_kind == "geohash":
            return GeohashCurriculumEnv(
                image=sample.image,
                ground_truth=ground_truth,
                renderer=renderer,
                config=env_config,
            )
        if cli.env_kind == "telescoping":
            return TelescopingGeoEnv(
                image=sample.image,
                ground_truth=ground_truth,
                renderer=renderer,
                config=telescoping_env_config,
            )
        raise ValueError(f"Unknown env_kind: {cli.env_kind}")

    # Training loop
    for step in range(cli.max_steps):
        t_start = time.time()

        # Get fresh sampling client
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"{step:06d}")

        # Get batch of samples
        samples: list[GeoSample] = []
        for _ in range(cli.batch_size):
            try:
                samples.append(next(sample_iter))
            except StopIteration:
                logger.info("Dataset exhausted, restarting...")
                sample_iter = iterate_samples(
                    hf_repo=cli.hf_repo,
                    max_shards=cli.max_shards,
                    seed=cli.seed + step,
                    local_path=cli.local_path,
                )
                samples.append(next(sample_iter))

        # Run episodes for each sample (with group_size copies for GRPO)
        training_datums: list[tinker.Datum] = []
        batch_rewards: list[float] = []
        batch_distances: list[float] = []
        skipped_uniform = 0

        for sample in samples:
            # Create group_size envs for this sample
            env_group = [create_env_from_sample(sample) for _ in range(cli.group_size)]

            group_rewards: list[float] = []
            group_transitions: list[list[tuple]] = []

            # Run episode for each env in group
            episode_results = await asyncio.gather(
                *[run_episode(env, sampling_client, sampling_params) for env in env_group]
            )

            for env, (transitions, metrics) in zip(env_group, episode_results):
                # Total reward across all turns
                episode_reward = sum(t[3] for t in transitions)
                group_rewards.append(episode_reward)
                group_transitions.append(transitions)

                if "distance/final_km" in metrics:
                    batch_distances.append(metrics["distance/final_km"])

            # Compute per-turn returns-to-go and center within the group (GRPO-style baseline).
            mean_reward = sum(group_rewards) / len(group_rewards)

            returns_to_go: list[list[float]] = []
            for transitions in group_transitions:
                rewards = [float(t[3]) for t in transitions]
                rtg: list[float] = []
                running = 0.0
                for r in reversed(rewards):
                    running += r
                    rtg.append(running)
                returns_to_go.append(list(reversed(rtg)))

            max_len = max((len(rtg) for rtg in returns_to_go), default=0)
            baseline_t: list[float] = []
            for t in range(max_len):
                vals = [rtg[t] for rtg in returns_to_go if len(rtg) > t]
                baseline_t.append(sum(vals) / len(vals))

            per_turn_advantages: list[list[float]] = []
            for rtg in returns_to_go:
                per_turn_advantages.append([rtg[t] - baseline_t[t] for t in range(len(rtg))])
            batch_rewards.append(mean_reward)

            # Skip if no within-group learning signal.
            if all(abs(a) < 1e-6 for advs in per_turn_advantages for a in advs):
                skipped_uniform += 1
                continue

            # Build training datums from all transitions
            for transitions, advantages_t in zip(group_transitions, per_turn_advantages):
                for (obs, tokens, logprobs, _turn_reward), advantage in zip(
                    transitions, advantages_t, strict=False
                ):
                    if not tokens or not logprobs:
                        continue

                    ob_len = obs.length - 1
                    full_chunks = list(obs.chunks) + [tinker.EncodedTextChunk(tokens=tokens[:-1])]
                    full_input = tinker.ModelInput(chunks=full_chunks)
                    full_targets = [0] * ob_len + tokens
                    all_logprobs = [0.0] * ob_len + logprobs
                    all_advantages = [0.0] * ob_len + [float(advantage)] * len(logprobs)

                    if full_input.length != len(full_targets):
                        continue

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
            logger.warning(f"Step {step}: no training datums")
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
            f"datums={len(training_datums)}, skipped={skipped_uniform}, time={elapsed:.1f}s"
        )

        if cli.wandb_project:
            wandb.log({
                "progress/step": step,
                "reward/mean": mean_reward,
                "distance_km/mean": mean_dist,
                "optim/datums": len(training_datums),
                "optim/skipped_uniform": skipped_uniform,
                "time/step_s": elapsed,
            })

        # Log to viz DB
        db.log_step(
            step=step,
            mean_reward=mean_reward,
            mean_distance_km=mean_dist,
            coord_tau=0,  # Not used in curriculum
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

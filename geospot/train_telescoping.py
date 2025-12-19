"""
Telescoping multi-turn GRPO training for geospot VLM.

Uses TelescopingGeoEnv: coarse -> refine -> final with potential-based rewards.
All turns predict coordinates. Reward = S(d_t) - S(d_{t-1}), so sum = final score.
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
from geospot.renderers import get_renderer, ensure_text
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.telescoping_geo_env import (
    TelescopingGeoEnv,
    TelescopingGeoEnvConfig,
)
from geospot.rl.geo_reward import GeoLocation

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI config for telescoping RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "sdan/geomix"
    max_shards: int | None = None
    max_steps: int = 100
    local_path: str | None = None  # Baseten cache: /root/.cache/user_artifacts/geomix/train

    # Training
    batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 4e-5
    max_tokens: int = 128
    temperature: float = 1.0

    # Telescoping config
    score_kind: str = "geoguessr"  # "geoguessr" or "exp"
    exp_tau_km: float = 2000.0

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-telescoping"

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
    env: TelescopingGeoEnv,
    sampling_client: tinker.SamplingClient,
    sampling_params: tinker.SamplingParams,
) -> tuple[list[tuple[tinker.ModelInput, list[int], list[float], float]], dict]:
    """Run a full multi-turn episode."""
    transitions = []
    final_metrics = {}

    obs, stop_condition = await env.initial_observation()

    while True:
        tokens, logprobs = await sample_single(sampling_client, obs, sampling_params)
        step_result = await env.step(tokens)

        transitions.append((obs, tokens, logprobs, step_result.reward))

        for k, v in step_result.metrics.items():
            if isinstance(v, (int, float)):
                final_metrics[k] = v

        if step_result.episode_done:
            break

        obs = step_result.next_observation

    return transitions, final_metrics


def create_env_from_sample(
    sample: GeoSample,
    renderer,
    env_config: TelescopingGeoEnvConfig,
) -> TelescopingGeoEnv:
    """Create TelescopingGeoEnv from a GeoSample."""
    ground_truth = GeoLocation(
        lat=sample.lat,
        lon=sample.lon,
        city=sample.city,
        country=sample.country,
    )
    return TelescopingGeoEnv(
        image=sample.image,
        ground_truth=ground_truth,
        renderer=renderer,
        config=env_config,
    )


async def run_training(cli: CLIConfig):
    """Main training loop for telescoping multi-turn."""
    if cli.log_path:
        log_path = cli.log_path
    else:
        model_name = cli.model_name.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_path = f"/tmp/geospot-telescoping/{model_name}-{timestamp}"

    check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"Telescoping GRPO Training: {cli.hf_repo} -> {log_path}")
    logger.info(f"Model: {cli.model_name}, batch={cli.batch_size}, group={cli.group_size}")
    logger.info(f"Telescoping: coarse -> refine -> final (potential-based rewards)")
    logger.info(f"Score kind: {cli.score_kind}")
    if cli.local_path:
        logger.info(f"Using local cache: {cli.local_path}")

    if cli.wandb_project:
        wandb.init(
            project=cli.wandb_project,
            name=f"telescoping-{cli.score_kind}",
            tags=["telescoping", "potential-shaping", "multi-turn"],
            config={
                "model_name": cli.model_name,
                "batch_size": cli.batch_size,
                "group_size": cli.group_size,
                "learning_rate": cli.learning_rate,
                "score_kind": cli.score_kind,
                "exp_tau_km": cli.exp_tau_km,
                "max_tokens": cli.max_tokens,
                "env_type": "telescoping",
            },
        )

    # Initialize components
    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    env_config = TelescopingGeoEnvConfig(
        score_kind=cli.score_kind,
        exp_tau_km=cli.exp_tau_km,
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

    # Viz DB
    run_id = str(uuid.uuid4())[:8]
    db = DBWriter(
        run_id=run_id,
        run_name=f"{cli.hf_repo}-telescoping",
        run_type="rl-telescoping",
        config={
            "model_name": cli.model_name,
            "hf_repo": cli.hf_repo,
            "lora_rank": cli.lora_rank,
            "batch_size": cli.batch_size,
            "group_size": cli.group_size,
            "learning_rate": cli.learning_rate,
            "max_steps": cli.max_steps,
            "score_kind": cli.score_kind,
        },
    )
    logger.info(f"Viz dashboard: http://localhost:3001/training-run/{run_id}")

    # Training loop
    for step in range(cli.max_steps):
        t_start = time.time()

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

        # Build env groups
        training_datums: list[tinker.Datum] = []
        batch_rewards: list[float] = []
        batch_distances: list[float] = []
        skipped_uniform = 0

        for sample in samples:
            env_group = [
                create_env_from_sample(sample, renderer, env_config)
                for _ in range(cli.group_size)
            ]

            episode_results = await asyncio.gather(
                *[run_episode(env, sampling_client, sampling_params) for env in env_group]
            )

            group_rewards: list[float] = []
            group_transitions: list[list[tuple]] = []

            for env, (transitions, metrics) in zip(env_group, episode_results):
                episode_reward = sum(t[3] for t in transitions)
                group_rewards.append(episode_reward)
                group_transitions.append(transitions)

                if "distance/final_km" in metrics:
                    batch_distances.append(metrics["distance/final_km"])

            # GRPO: group-centered advantages
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]
            batch_rewards.append(mean_reward)

            if all(abs(a) < 1e-6 for a in advantages):
                skipped_uniform += 1
                continue

            # Build training datums
            for transitions, advantage in zip(group_transitions, advantages):
                for obs, tokens, logprobs, turn_reward in transitions:
                    if not tokens or not logprobs:
                        continue

                    ob_len = obs.length - 1
                    full_chunks = list(obs.chunks) + [tinker.EncodedTextChunk(tokens=tokens[:-1])]
                    full_input = tinker.ModelInput(chunks=full_chunks)
                    full_targets = [0] * ob_len + tokens
                    all_logprobs = [0.0] * ob_len + logprobs
                    all_advantages = [0.0] * ob_len + [advantage] * len(logprobs)

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

        db.log_step(
            step=step,
            mean_reward=mean_reward,
            mean_distance_km=mean_dist,
            coord_tau=0,
            num_datums=len(training_datums),
            elapsed_s=elapsed,
        )

        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved checkpoint: step_{step:06d}")

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

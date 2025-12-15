"""
GRPO training for geospot VLM.
"""

import asyncio
import logging
import os
import time
from datetime import datetime

import chz
import tinker
import torch
import wandb
from tinker import TensorData
from tinker.types import AdamParams

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.renderers import get_renderer
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.geo_dataset import StreamingGeoDataset
from geospot.rl.geo_env import GeoEnv, GeoEnvConfig
from geospot.rl.geo_reward import GeoRewardConfig

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for geospot RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "sdan/geospot-unified"
    max_shards: int | None = None
    max_steps: int = 100

    # Training
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    max_tokens: int = 256
    temperature: float = 1.0

    # Reward
    coord_tau: float = 25.0
    coord_weight: float = 0.7

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


def main(cli: CLIConfig):
    """GRPO training loop for geo VLM."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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

    if cli.wandb_project:
        wandb.init(
            project=cli.wandb_project,
            config={
                "model_name": cli.model_name,
                "batch_size": cli.batch_size,
                "group_size": cli.group_size,
                "learning_rate": cli.learning_rate,
                "coord_tau": cli.coord_tau,
                "coord_weight": cli.coord_weight,
                "max_tokens": cli.max_tokens,
                "temperature": cli.temperature,
            },
        )

    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    reward_config = GeoRewardConfig(coord_tau=cli.coord_tau, coord_weight=cli.coord_weight)
    env_config = GeoEnvConfig(reward_config=reward_config)

    dataset = StreamingGeoDataset(
        hf_repo=cli.hf_repo,
        group_size=cli.group_size,
        renderer=renderer,
        env_config=env_config,
        max_shards=cli.max_shards,
        seed=cli.seed,
    )

    service_client = tinker.ServiceClient(base_url=cli.base_url)
    if cli.load_checkpoint_path:
        training_client = service_client.create_training_client_from_state(cli.load_checkpoint_path)
        logger.info(f"Loaded checkpoint from {cli.load_checkpoint_path}")
    else:
        training_client = service_client.create_lora_training_client(
            cli.model_name, rank=cli.lora_rank
        )

    sampling_params = tinker.SamplingParams(
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = AdamParams(learning_rate=cli.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    for step in range(cli.max_steps):
        t_start = time.time()

        # Get fresh sampling client each step
        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Get batch of env group builders
        builders = dataset.get_batch(cli.batch_size)
        if not builders:
            continue

        # Create envs from builders (one env per builder for now, group sampling happens below)
        loop = asyncio.new_event_loop()
        envs: list[GeoEnv] = []
        for builder in builders:
            env_group = loop.run_until_complete(builder.make_envs())
            envs.append(env_group[0])  # Take first env from group

        training_datums: list[tinker.Datum] = []
        batch_rewards: list[float] = []
        batch_distances: list[float] = []

        # Step 1: Get all observations
        env_obs: list[tuple[tinker.ModelInput, GeoEnv]] = []
        for env in envs:
            ob, _ = loop.run_until_complete(env.initial_observation())
            env_obs.append((ob, env))

        # Step 2: Fire off ALL sample requests in parallel (non-blocking)
        all_futures: list[list] = []
        for ob, env in env_obs:
            group_futures = []
            for _ in range(cli.group_size):
                future = sampling_client.sample(
                    prompt=ob,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                group_futures.append(future)
            all_futures.append(group_futures)

        # Step 3: Collect results and compute rewards
        for (ob, env), group_futures in zip(env_obs, all_futures):
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_rewards: list[float] = []
            group_distances: list[float] = []

            for future in group_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                group_tokens.append(sampled_tokens)
                group_logprobs.append(sampled_logprobs)

                # Compute reward
                sample_env = GeoEnv(
                    image=env.image,
                    ground_truth=env.ground_truth,
                    renderer=env.renderer,
                    config=env.config,
                )
                step_result = loop.run_until_complete(sample_env.step(sampled_tokens))
                group_rewards.append(step_result.reward)
                if "distance_km" in step_result.metrics:
                    group_distances.append(step_result.metrics["distance_km"])

            # Compute group-centered advantages
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]
            batch_rewards.append(mean_reward)
            if group_distances:
                batch_distances.append(sum(group_distances) / len(group_distances))

            # Skip if all advantages are zero (no gradient signal)
            if all(a == 0.0 for a in advantages):
                continue

            # Build training datums
            ob_len = ob.length - 1
            for tokens, logprobs, advantage in zip(group_tokens, group_logprobs, advantages):
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

        loop.close()

        if not training_datums:
            logger.warning(f"Step {step}: no training datums (all uniform rewards)")
            continue

        # Training step
        fwd_bwd_future = training_client.forward_backward(training_datums, loss_fn="importance_sampling")
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_future.result()
        optim_future.result()

        # Logging
        mean_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
        mean_dist = sum(batch_distances) / len(batch_distances) if batch_distances else 0
        elapsed = time.time() - t_start
        logger.info(
            f"Step {step}: reward={mean_reward:.3f}, dist={mean_dist:.0f}km, "
            f"datums={len(training_datums)}, time={elapsed:.1f}s"
        )

        if cli.wandb_project:
            wandb.log({
                "reward": mean_reward,
                "distance_km": mean_dist,
                "datums": len(training_datums),
                "time_s": elapsed,
                "step": step,
            })

        # Save checkpoint
        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved checkpoint: step_{step:06d}")

    # Final checkpoint
    result = training_client.save_state(name="final").result()
    logger.info("GRPO training complete!")
    logger.info(f"Checkpoint: {result.path}")

    if cli.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    chz.nested_entrypoint(main)

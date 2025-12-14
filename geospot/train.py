"""
Training script for geospot VLM.

Usage:
    python -m geospot.train \
        hf_repo=sdan/geospot-data \
        model_name=Qwen/Qwen3-VL-235B-A22B-Instruct \
        log_path=./runs/exp1
"""

import asyncio
import logging
import os
from datetime import datetime

import chz
import tinker
from tinker.types import LossFnType

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.rl import GeoDatasetBuilder, GeoRewardConfig

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for geospot training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str  # Required: HuggingFace dataset
    hf_split: str = "train"
    hf_test_split: str | None = "test"

    # Training
    batch_size: int = 16
    group_size: int = 4
    learning_rate: float = 5e-4
    max_tokens: int = 256
    temperature: float = 1.0
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"

    # Reward
    coord_tau: float = 25.0
    coord_weight: float = 0.7

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing
    save_every: int = 100
    eval_every: int = 50

    # Misc
    seed: int = 0
    base_url: str | None = None
    behavior_if_log_dir_exists: LogdirBehavior = "ask"


async def main(cli: CLIConfig):
    """Main training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build log path
    if cli.log_path:
        log_path = cli.log_path
    else:
        model_name = cli.model_name.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_path = f"/tmp/geospot/{model_name}-{timestamp}"

    check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"Log path: {log_path}")
    logger.info(f"Dataset: {cli.hf_repo}")
    logger.info(f"Model: {cli.model_name}")

    # Build dataset
    dataset_builder = GeoDatasetBuilder(
        hf_repo=cli.hf_repo,
        hf_split=cli.hf_split,
        hf_test_split=cli.hf_test_split,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        model_name_for_tokenizer=cli.model_name,
        renderer_name=cli.renderer_name,
        seed=cli.seed,
        reward_config=GeoRewardConfig(
            coord_tau=cli.coord_tau,
            coord_weight=cli.coord_weight,
        ),
    )

    train_dataset, test_dataset = await dataset_builder()
    logger.info(f"Train batches: {len(train_dataset)}")
    if test_dataset:
        logger.info(f"Test batches: {len(test_dataset)}")

    # Initialize tinker client
    service_client = tinker.ServiceClient(base_url=cli.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cli.model_name, rank=cli.lora_rank
    )

    # Training loop
    # Import from tinker_cookbook for full implementation
    # For now, this is a minimal skeleton
    from geospot.rl import Transition, Trajectory, TrajectoryGroup
    from geospot.completers import TinkerTokenCompleter

    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
    )

    num_batches = len(train_dataset)
    for i_batch in range(num_batches):
        logger.info(f"Batch {i_batch + 1}/{num_batches}")

        # Get batch of env builders
        env_builders = train_dataset.get_batch(i_batch)
        if not env_builders:
            continue

        # Collect rollouts
        all_rewards = []
        for builder in env_builders:
            envs = await builder.make_envs()
            for env in envs:
                ob, stop = await env.initial_observation()
                action_result = await completer(ob, stop)
                result = await env.step(action_result.tokens)
                all_rewards.append(result.reward)

        # Log metrics
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        logger.info(f"  Mean reward: {mean_reward:.3f}")

        # Save checkpoint
        if cli.save_every > 0 and (i_batch + 1) % cli.save_every == 0:
            logger.info(f"  Saving checkpoint at batch {i_batch + 1}")
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            completer = TinkerTokenCompleter(
                sampling_client=sampling_client,
                max_tokens=cli.max_tokens,
                temperature=cli.temperature,
            )

    logger.info("Training complete!")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))

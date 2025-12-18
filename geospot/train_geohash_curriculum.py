"""
Geohash curriculum training using master-tinker's RL pipeline.

Uses the proper multi-turn handling from tinker_cookbook.rl.train.
"""

import asyncio
import logging
import sys

# Add master-tinker to path
sys.path.insert(0, "/Users/sdan/Developer/master-tinker")

import chz
from tinker_cookbook.rl import train as rl_train

from geospot.rl.geo_dataset import GeohashCurriculumDatasetBuilder
from geospot.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class GeohashCurriculumTrainConfig:
    """Training config for geohash curriculum."""

    # Data
    hf_repo: str = "sdan/geomix"
    max_samples: int = 5000

    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32

    # Training
    batch_size: int = 32
    group_size: int = 8
    learning_rate: float = 4e-5
    max_tokens: int = 256  # Per turn
    temperature: float = 1.0

    # Curriculum config
    improvement_bonus: float = 0.1

    # Logging
    log_path: str = "/tmp/geospot-geohash-curriculum"
    wandb_project: str | None = "geospot-curriculum"

    # Checkpointing
    save_every: int = 25
    eval_every: int = 25


def build_dataset_builder(cfg: GeohashCurriculumTrainConfig) -> RLDatasetBuilder:
    """Create the dataset builder for geohash curriculum training."""
    return GeohashCurriculumDatasetBuilder(
        hf_repo=cfg.hf_repo,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name="qwen3_vl",
        max_samples=cfg.max_samples,
        improvement_bonus=cfg.improvement_bonus,
    )


def main(cfg: GeohashCurriculumTrainConfig):
    """Run geohash curriculum training using master-tinker pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting Geohash Curriculum Training")
    logger.info(f"  Model: {cfg.model_name}")
    logger.info(f"  Dataset: {cfg.hf_repo}")
    logger.info(f"  Batch size: {cfg.batch_size}, Group size: {cfg.group_size}")
    logger.info(f"  Curriculum: 3-turn (coarse→medium→fine)")

    # Build the RL training config using master-tinker's Config
    rl_config = rl_train.Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=build_dataset_builder(cfg),
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        lora_rank=cfg.lora_rank,
        log_path=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=f"geohash-curriculum-{cfg.hf_repo.split('/')[-1]}",
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        remove_constant_reward_groups=True,
    )

    # Run training using master-tinker's battle-tested pipeline
    asyncio.run(rl_train.main(rl_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

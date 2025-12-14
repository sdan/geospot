"""
Supervised fine-tuning for geospot VLM warm-start.

Usage (WebDataset streaming from HF):
    python -m geospot.sft \
        hf_repo=sdan/geospot-vista9 \
        log_path=./runs/sft-warmstart

Then use the checkpoint for RL:
    python -m geospot.train \
        load_checkpoint_path=./runs/sft-warmstart/checkpoints/final \
        log_path=./runs/rl-from-sft
"""

import asyncio
import logging
import os
import random
from datetime import datetime
from typing import Any, Iterator

import chz
import tinker
import webdataset as wds
from huggingface_hub import list_repo_files
from PIL import Image

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.renderers import (
    ImagePart,
    Message,
    TextPart,
    TrainOnWhat,
    get_renderer,
)
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.geo_env import DEFAULT_GEO_PROMPT

logger = logging.getLogger(__name__)


def format_ground_truth(lat: float, lon: float, city: str | None, country: str | None) -> str:
    """Format ground truth as the target response."""
    lines = []
    if city:
        lines.append(f"City: {city}")
    if country:
        lines.append(f"Country: {country}")
    lines.append(f"Latitude: {lat:.4f}")
    lines.append(f"Longitude: {lon:.4f}")
    return "\n".join(lines)


def get_shard_urls(hf_repo: str, max_shards: int | None = None, seed: int = 0, prefix: str = "shardheadings/") -> list[str]:
    """Get shuffled tar shard URLs from HuggingFace."""
    logger.info(f"Fetching shard list from {hf_repo}...")
    files = list(list_repo_files(hf_repo, repo_type="dataset"))

    # Filter to shardheadings/ directory
    tar_files = [f for f in files if f.endswith(".tar") and f.startswith(prefix)]
    logger.info(f"Found {len(tar_files)} shards in {prefix}")

    # Shuffle shards
    random.seed(seed)
    random.shuffle(tar_files)

    if max_shards:
        tar_files = tar_files[:max_shards]

    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    urls = [f"{base_url}/{f}" for f in tar_files]
    logger.info(f"Using {len(urls)} shards")
    return urls


class WebDatasetIterator:
    """Streaming iterator over WebDataset shards."""

    def __init__(
        self,
        urls: list[str],
        renderer,
        max_length: int,
        train_on_what: TrainOnWhat,
        shuffle_buffer: int = 1000,
    ):
        self.urls = urls
        self.renderer = renderer
        self.max_length = max_length
        self.train_on_what = train_on_what
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[tinker.Datum]:
        dataset = (
            wds.WebDataset(self.urls, shardshuffle=True)
            .shuffle(self.shuffle_buffer)
            .decode("pil")
        )

        for sample in dataset:
            datum = self._sample_to_datum(sample)
            if datum is not None:
                yield datum

    def _get_image(self, sample: dict[str, Any]) -> Image.Image | None:
        """Get image from sample, handling different formats."""
        # Single image format (country shards)
        if "jpg" in sample:
            return sample["jpg"]
        if "png" in sample:
            return sample["png"]
        # Multi-heading format (shardheadings/) - pick random heading
        headings = ["000.jpg", "090.jpg", "180.jpg", "270.jpg"]
        available = [h for h in headings if h in sample]
        if available:
            return sample[random.choice(available)]
        return None

    def _sample_to_datum(self, sample: dict[str, Any]) -> tinker.Datum | None:
        try:
            image = self._get_image(sample)
            if image is None:
                return None
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            json_data = sample.get("json", {})
            lat = float(json_data.get("lat", 0))
            lon = float(json_data.get("lng") or json_data.get("lon", 0))
            # Handle different metadata formats
            city = json_data.get("city")
            country = json_data.get("country") or json_data.get("region")

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return None

            # Build conversation
            user_content = [
                ImagePart(type="image", image=image),
                TextPart(type="text", text=DEFAULT_GEO_PROMPT),
            ]
            assistant_content = format_ground_truth(lat, lon, city, country)

            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content),
            ]

            model_input, weights = self.renderer.build_supervised_example(
                messages, train_on_what=self.train_on_what
            )

            if self.max_length and model_input.length > self.max_length:
                return None

            return tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={"weights": tinker.TensorData.from_numpy(weights.numpy())},
            )
        except Exception as e:
            logger.debug(f"Failed to create datum: {e}")
            return None


@chz.chz
class CLIConfig:
    """CLI config for SFT warm-start."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "sdan/geospot-vista9"
    max_shards: int | None = None  # None = all shards
    max_steps: int = 1000

    # Training
    batch_size: int = 16
    learning_rate: float = 5e-4
    max_length: int = 4096
    shuffle_buffer: int = 1000

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None

    # Checkpointing
    save_every: int = 100

    # Misc
    seed: int = 0
    base_url: str | None = None
    behavior_if_log_dir_exists: LogdirBehavior = "ask"


async def main(cli: CLIConfig):
    """SFT training loop with WebDataset streaming."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if cli.log_path:
        log_path = cli.log_path
    else:
        model_name = cli.model_name.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_path = f"/tmp/geospot-sft/{model_name}-{timestamp}"

    check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"SFT warm-start: {cli.hf_repo} -> {log_path}")
    logger.info(f"Model: {cli.model_name}, batch_size={cli.batch_size}, max_steps={cli.max_steps}")

    # Setup renderer
    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    # Get shard URLs
    urls = get_shard_urls(cli.hf_repo, max_shards=cli.max_shards, seed=cli.seed)

    # Create streaming iterator
    data_iter = iter(WebDatasetIterator(
        urls=urls,
        renderer=renderer,
        max_length=cli.max_length,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        shuffle_buffer=cli.shuffle_buffer,
    ))

    # Initialize client
    service_client = tinker.ServiceClient(base_url=cli.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cli.model_name, rank=cli.lora_rank
    )

    # Training loop
    for step in range(cli.max_steps):
        # Collect batch
        batch = []
        while len(batch) < cli.batch_size:
            try:
                datum = next(data_iter)
                batch.append(datum)
            except StopIteration:
                logger.info("Dataset exhausted, reshuffling...")
                urls = get_shard_urls(cli.hf_repo, max_shards=cli.max_shards, seed=cli.seed + step)
                data_iter = iter(WebDatasetIterator(
                    urls=urls,
                    renderer=renderer,
                    max_length=cli.max_length,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    shuffle_buffer=cli.shuffle_buffer,
                ))

        if not batch:
            logger.warning("Empty batch, skipping step")
            continue

        # LR schedule (linear decay)
        lr = cli.learning_rate * (1 - step / cli.max_steps)

        fwd_bwd = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        optim = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))

        result = await fwd_bwd.result_async()
        await optim.result_async()

        # Log
        num_tokens = sum(d.model_input.length for d in batch)
        loss = result.loss if hasattr(result, 'loss') else 0
        logger.info(f"Step {step}: {len(batch)} seqs, {num_tokens} tokens, loss={loss:.4f}, lr={lr:.2e}")

        # Checkpoint
        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            name = f"step_{step:06d}"
            await training_client.save_weights_async(name)
            logger.info(f"Saved checkpoint: {name}")

    # Final checkpoint
    await training_client.save_weights_async("final")
    logger.info(f"SFT complete. Final checkpoint saved.")


def cli_main():
    """CLI entry point for geospot-sft command."""
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))


if __name__ == "__main__":
    cli_main()

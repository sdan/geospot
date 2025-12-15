"""
SFT warm-start for geospot VLM.
"""

import logging
import os
import time
from datetime import datetime
from typing import Iterator

import chz
import tinker
import torch

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.data import GeoSample, get_shard_urls, iterate_samples
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


def _create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """Create input/target split for next-token prediction."""
    assert len(chunks) >= 1
    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.types.EncodedTextChunk):
        raise ValueError("Last chunk must be text")

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise ValueError("need at least 2 tokens")

    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.types.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


def datum_from_model_input_weights(
    model_input: tinker.ModelInput,
    weights: torch.Tensor,
    max_length: int | None = None,
) -> tinker.Datum | None:
    """Create Datum with proper target_tokens and weights formatting."""
    model_input_chunks = list(model_input.chunks)

    # Truncate to max_length
    if max_length is not None:
        total_length = sum(chunk.length for chunk in model_input_chunks)
        while total_length > max_length and model_input_chunks:
            last = model_input_chunks[-1]
            if isinstance(last, tinker.types.EncodedTextChunk):
                overflow = total_length - max_length
                if overflow < last.length:
                    model_input_chunks[-1] = tinker.types.EncodedTextChunk(
                        tokens=list(last.tokens[:-overflow])
                    )
                    total_length = max_length
                else:
                    model_input_chunks.pop()
                    total_length -= last.length
            else:
                model_input_chunks.pop()
                total_length -= last.length

    # Remove trailing images
    while model_input_chunks and isinstance(
        model_input_chunks[-1], (tinker.types.ImageChunk, tinker.types.ImageAssetPointerChunk)
    ):
        model_input_chunks.pop()

    if not model_input_chunks:
        return None

    input_model_input, target_tokens = _create_rightshifted_model_input_and_leftshifted_targets(
        model_input_chunks
    )
    weights = weights[1 : len(target_tokens) + 1]

    return tinker.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )


def format_ground_truth(lat: float, lon: float) -> str:
    """Format ground truth as the target response."""
    return f"Latitude: {lat:.6f}\nLongitude: {lon:.6f}"


def sample_to_datum(
    sample: GeoSample,
    renderer,
    max_length: int,
    train_on_what: TrainOnWhat,
) -> tinker.Datum | None:
    """Convert a GeoSample to a training Datum."""
    try:
        user_content = [
            ImagePart(type="image", image=sample.image),
            TextPart(type="text", text=DEFAULT_GEO_PROMPT),
        ]
        assistant_content = format_ground_truth(sample.lat, sample.lon)

        messages = [
            Message(role="user", content=user_content),
            Message(role="assistant", content=assistant_content),
        ]

        model_input, weights = renderer.build_supervised_example(
            messages, train_on_what=train_on_what
        )

        return datum_from_model_input_weights(model_input, weights, max_length=max_length)
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
    hf_repo: str = "sdan/geospot-unified"
    max_shards: int | None = None
    max_steps: int = 1000

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-4
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


def main(cli: CLIConfig):
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

    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    def make_sample_iter(seed: int):
        urls = get_shard_urls(cli.hf_repo, max_shards=cli.max_shards, seed=seed)
        return iterate_samples(urls, shuffle_buffer=cli.shuffle_buffer)

    sample_iter = make_sample_iter(cli.seed)

    service_client = tinker.ServiceClient(base_url=cli.base_url)
    training_client = service_client.create_lora_training_client(
        cli.model_name, rank=cli.lora_rank
    )

    for step in range(cli.max_steps):
        t_start = time.time()

        # Collect batch
        batch = []
        while len(batch) < cli.batch_size:
            try:
                sample = next(sample_iter)
                datum = sample_to_datum(
                    sample,
                    renderer=renderer,
                    max_length=cli.max_length,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                if datum is not None:
                    batch.append(datum)
            except StopIteration:
                logger.info("Dataset exhausted, reshuffling...")
                sample_iter = make_sample_iter(cli.seed + step)

        if not batch:
            logger.warning("Empty batch, skipping step")
            continue

        # LR schedule (linear decay)
        lr = cli.learning_rate * (1 - step / cli.max_steps)

        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=lr))
        result = fwd_bwd_future.result()
        optim_future.result()

        # Log
        num_tokens = sum(d.model_input.length for d in batch)
        elapsed = time.time() - t_start
        logger.info(f"Step {step}: tokens={num_tokens}, lr={lr:.2e}, time={elapsed:.1f}s")

        # Checkpoint
        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved checkpoint: step_{step:06d}")

    # Final checkpoint
    result = training_client.save_state(name="final").result()
    logger.info("SFT complete!")
    logger.info(f"Checkpoint: {result.path}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)

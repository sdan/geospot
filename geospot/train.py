"""
GRPO training for geospot VLM.
"""

import asyncio
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Iterator

import chz
import tinker
import torch
import webdataset as wds
from huggingface_hub import list_repo_files
from PIL import Image
from tinker import TensorData
from tinker.types import AdamParams

from geospot.cli_utils import check_log_dir, LogdirBehavior
from geospot.renderers import get_renderer
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor
from geospot.rl.geo_env import GeoEnv, GeoEnvConfig
from geospot.rl.geo_reward import GeoLocation, GeoRewardConfig

logger = logging.getLogger(__name__)


def get_shard_urls(hf_repo: str, max_shards: int | None = None, seed: int = 0, prefix: str = "shardheadings/") -> list[str]:
    """Get shuffled tar shard URLs from HuggingFace."""
    logger.info(f"Fetching shard list from {hf_repo}...")
    files = list(list_repo_files(hf_repo, repo_type="dataset"))
    tar_files = [f for f in files if f.endswith(".tar") and f.startswith(prefix)]
    logger.info(f"Found {len(tar_files)} shards in {prefix}")

    random.seed(seed)
    random.shuffle(tar_files)

    if max_shards:
        tar_files = tar_files[:max_shards]

    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    urls = [f"{base_url}/{f}" for f in tar_files]
    logger.info(f"Using {len(urls)} shards")
    return urls


class WebDatasetEnvIterator:
    """Streaming iterator that yields GeoEnv instances from WebDataset."""

    def __init__(
        self,
        urls: list[str],
        renderer,
        env_config: GeoEnvConfig,
        shuffle_buffer: int = 1000,
        max_image_size: int = 480,
    ):
        self.urls = urls
        self.renderer = renderer
        self.env_config = env_config
        self.shuffle_buffer = shuffle_buffer
        self.max_image_size = max_image_size

    def __iter__(self) -> Iterator[GeoEnv]:
        def warn_and_continue(exn):
            logger.warning(f"Skipping shard: {exn}")
            return True

        dataset = (
            wds.WebDataset(self.urls, shardshuffle=1000, handler=warn_and_continue)
            .shuffle(self.shuffle_buffer)
            .decode("pil", handler=warn_and_continue)
        )

        for sample in dataset:
            env = self._sample_to_env(sample)
            if env is not None:
                yield env

    def _get_image(self, sample: dict[str, Any]) -> Image.Image | None:
        if "jpg" in sample:
            return sample["jpg"]
        if "png" in sample:
            return sample["png"]
        headings = ["000.jpg", "090.jpg", "180.jpg", "270.jpg"]
        available = [h for h in headings if h in sample]
        if available:
            return sample[random.choice(available)]
        return None

    def _sample_to_env(self, sample: dict[str, Any]) -> GeoEnv | None:
        try:
            image = self._get_image(sample)
            if image is None:
                return None
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            # Resize if needed
            if self.max_image_size and max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)

            json_data = sample.get("json", {})
            lat = float(json_data.get("lat", 0))
            lon = float(json_data.get("lng") or json_data.get("lon", 0))

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return None

            ground_truth = GeoLocation(
                lat=lat,
                lon=lon,
                city=json_data.get("city"),
                country=json_data.get("country") or json_data.get("region"),
            )

            return GeoEnv(
                image=image,
                ground_truth=ground_truth,
                renderer=self.renderer,
                config=self.env_config,
            )
        except Exception as e:
            logger.debug(f"Failed to create env: {e}")
            return None


@chz.chz
class CLIConfig:
    """Command-line configuration for geospot RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "sdan/geospot-vista9"
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

    tokenizer = get_tokenizer(cli.model_name)
    image_processor = get_image_processor(cli.model_name)
    renderer = get_renderer(cli.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    reward_config = GeoRewardConfig(coord_tau=cli.coord_tau, coord_weight=cli.coord_weight)
    env_config = GeoEnvConfig(reward_config=reward_config)

    urls = get_shard_urls(cli.hf_repo, max_shards=cli.max_shards, seed=cli.seed)
    env_iter = iter(WebDatasetEnvIterator(
        urls=urls,
        renderer=renderer,
        env_config=env_config,
        shuffle_buffer=1000,
    ))

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

        # Collect batch of envs
        envs: list[GeoEnv] = []
        while len(envs) < cli.batch_size:
            try:
                envs.append(next(env_iter))
            except StopIteration:
                logger.info("Dataset exhausted, reshuffling...")
                urls = get_shard_urls(cli.hf_repo, max_shards=cli.max_shards, seed=cli.seed + step)
                env_iter = iter(WebDatasetEnvIterator(
                    urls=urls, renderer=renderer, env_config=env_config, shuffle_buffer=1000,
                ))

        if not envs:
            continue

        training_datums: list[tinker.Datum] = []
        batch_rewards: list[float] = []
        batch_distances: list[float] = []

        # Step 1: Get all observations
        loop = asyncio.new_event_loop()
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

        # Save checkpoint
        if cli.save_every > 0 and step > 0 and step % cli.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved checkpoint: step_{step:06d}")

    # Final checkpoint
    result = training_client.save_state(name="final").result()
    logger.info("GRPO training complete!")
    logger.info(f"Checkpoint: {result.path}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)

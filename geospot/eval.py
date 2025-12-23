"""
Evaluation utilities for geospot training.

Runs inference on held-out test set and computes distance/score metrics.
Uses async batching for efficient parallel sampling.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterator

import tinker

from geospot.datasets import GeoSample, iterate_samples
from geospot.envs import (
    GeoLocation,
    SingleTurnGeoEnv,
    SingleTurnGeoEnvConfig,
    geoguessr_score,
    haversine_km,
    parse_geo_response,
    SINGLE_TURN_PROMPT,
)
from geospot.renderers import Renderer, ensure_text, ImagePart, TextPart, Message

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from running evaluation."""
    mean_distance_km: float
    mean_score: float
    median_distance_km: float
    acc_1km: float      # % within 1km
    acc_25km: float     # % within 25km
    acc_200km: float    # % within 200km
    acc_750km: float    # % within 750km
    num_samples: int
    num_failed: int     # Failed to parse


async def run_eval(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    hf_repo: str = "osv5m/osv5m",
    split: str = "test",
    num_samples: int = 100,
    max_tokens: int = 128,
    seed: int = 42,
    max_parallel: int = 32,  # Concurrent sampling limit
) -> EvalResult:
    """
    Run evaluation on test set with efficient async batching.

    Args:
        sampling_client: Tinker sampling client with current model weights
        renderer: Message renderer
        hf_repo: HuggingFace dataset repo
        split: Dataset split to use ("test")
        num_samples: Number of samples to evaluate
        max_tokens: Max tokens to generate
        seed: Random seed for reproducibility
        max_parallel: Max concurrent sampling requests

    Returns:
        EvalResult with distance/score metrics
    """
    # Load test samples
    sample_iter = iterate_samples(
        hf_repo=hf_repo,
        split=split,
        seed=seed,
        shuffle_buffer=100,
        max_image_size=512,
    )

    # Collect samples first
    samples: list[GeoSample] = []
    for i, sample in enumerate(sample_iter):
        if i >= num_samples:
            break
        samples.append(sample)

    if not samples:
        return EvalResult(
            mean_distance_km=float("inf"), mean_score=0.0, median_distance_km=float("inf"),
            acc_1km=0.0, acc_25km=0.0, acc_200km=0.0, acc_750km=0.0,
            num_samples=0, num_failed=0,
        )

    # Build prompts for all samples
    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker.SamplingParams(
        stop=stop_sequences,
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy for eval
    )

    # Concurrency limiter
    semaphore = asyncio.Semaphore(max_parallel)

    async def sample_one(sample: GeoSample) -> tuple[GeoSample, str | None]:
        """Sample prediction for one image, return (sample, response_text or None)."""
        async with semaphore:
            try:
                # Build prompt directly (simpler than using env)
                messages = [Message(role="user", content=[
                    ImagePart(type="image", image=sample.image),
                    TextPart(type="text", text=SINGLE_TURN_PROMPT),
                ])]
                prompt = renderer.build_generation_prompt(messages)

                result = await sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                return sample, renderer.tokenizer.decode(result.sequences[0].tokens)
            except Exception as e:
                logger.debug(f"Eval sample failed: {e}")
                return sample, None

    # Run all samples in parallel (bounded by semaphore)
    tasks = [asyncio.create_task(sample_one(s)) for s in samples]
    results = await asyncio.gather(*tasks)

    # Process results
    distances: list[float] = []
    scores: list[float] = []
    num_failed = 0

    for sample, response_text in results:
        if response_text is None:
            num_failed += 1
            continue

        parsed = parse_geo_response(response_text)
        if parsed.location is None:
            num_failed += 1
            continue

        dist = haversine_km(
            parsed.location.lat, parsed.location.lon,
            sample.lat, sample.lon
        )
        distances.append(dist)
        scores.append(geoguessr_score(dist))

    if not distances:
        return EvalResult(
            mean_distance_km=float("inf"), mean_score=0.0, median_distance_km=float("inf"),
            acc_1km=0.0, acc_25km=0.0, acc_200km=0.0, acc_750km=0.0,
            num_samples=0, num_failed=num_failed,
        )

    # Compute statistics
    distances_sorted = sorted(distances)
    n = len(distances)

    return EvalResult(
        mean_distance_km=sum(distances) / n,
        mean_score=sum(scores) / n,
        median_distance_km=distances_sorted[n // 2],
        acc_1km=sum(1 for d in distances if d < 1) / n,
        acc_25km=sum(1 for d in distances if d < 25) / n,
        acc_200km=sum(1 for d in distances if d < 200) / n,
        acc_750km=sum(1 for d in distances if d < 750) / n,
        num_samples=n,
        num_failed=num_failed,
    )


def eval_result_to_dict(result: EvalResult, prefix: str = "eval") -> dict:
    """Convert EvalResult to dict for wandb logging."""
    return {
        f"{prefix}/distance_km": result.mean_distance_km,
        f"{prefix}/score": result.mean_score,
        f"{prefix}/median_distance_km": result.median_distance_km,
        f"{prefix}/acc_1km": result.acc_1km,
        f"{prefix}/acc_25km": result.acc_25km,
        f"{prefix}/acc_200km": result.acc_200km,
        f"{prefix}/acc_750km": result.acc_750km,
        f"{prefix}/num_samples": result.num_samples,
        f"{prefix}/num_failed": result.num_failed,
    }

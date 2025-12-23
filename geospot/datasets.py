"""
Dataset loading for geolocation training.

Uses HuggingFace datasets streaming - no manifests, no fallbacks, just:
    load_dataset(..., streaming=True)

Reference: tinker_cookbook/recipes/vlm_classifier/data.py
"""

import logging
import random
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GeoSample:
    """A geolocation sample with image and coordinates + OSV-5M labels."""
    image: Image.Image
    lat: float
    lon: float
    # OSV-5M hierarchical labels
    country: str | None = None
    region: str | None = None
    sub_region: str | None = None
    city: str | None = None
    source: str | None = None


def iterate_samples(
    hf_repo: str = "osv5m/osv5m",
    split: str = "train",
    seed: int = 0,
    shuffle_buffer: int = 1000,
    max_image_size: int = 512,
) -> Iterator[GeoSample]:
    """
    Stream GeoSamples from HuggingFace dataset.

    Args:
        hf_repo: HuggingFace dataset repo (e.g., "osv5m/osv5m")
        split: Dataset split ("train", "test", etc.)
        seed: Random seed for shuffling
        shuffle_buffer: Buffer size for streaming shuffle
        max_image_size: Resize images to this size (shortest side)

    Yields:
        GeoSample objects with image, lat, lon, and metadata
    """
    logger.info(f"Loading {hf_repo} ({split}) with streaming=True...")

    ds = load_dataset(hf_repo, split=split, streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for idx, sample in enumerate(ds):
        try:
            image = sample.get("image")
            if image is None:
                continue

            # Ensure PIL Image
            if not isinstance(image, Image.Image):
                continue

            # Convert to RGB
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            # Resize if needed
            if max_image_size:
                image = _resize_and_crop(image, max_image_size)

            lat = float(sample.get("latitude", 0))
            lon = float(sample.get("longitude", 0))

            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            if idx == 0:
                logger.info(f"First sample received from {hf_repo}")

            yield GeoSample(
                image=image,
                lat=lat,
                lon=lon,
                country=sample.get("country"),
                region=sample.get("region"),
                sub_region=sample.get("sub-region"),  # Note: hyphen in OSV-5M
                city=sample.get("city"),
                source=hf_repo,
            )
        except Exception as e:
            logger.debug(f"Skipping sample: {e}")
            continue


def _resize_and_crop(image: Image.Image, target_size: int) -> Image.Image:
    """Resize shortest side to target_size, then random crop to square."""
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Random crop to square
    left = random.randint(0, max(0, new_w - target_size))
    top = random.randint(0, max(0, new_h - target_size))
    return image.crop((left, top, left + target_size, top + target_size))

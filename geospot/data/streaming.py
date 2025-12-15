"""
Streaming data loading for geospot-unified dataset.

Uses webdataset for efficient streaming without resolving all files upfront.
"""

import logging
import random
from typing import Iterator

import webdataset as wds
from huggingface_hub import list_repo_files
from PIL import Image

logger = logging.getLogger(__name__)


class GeoSample:
    """A geolocation sample with image and coordinates."""

    def __init__(
        self,
        image: Image.Image,
        lat: float,
        lon: float,
        city: str | None = None,
        country: str | None = None,
        source: str | None = None,
    ):
        self.image = image
        self.lat = lat
        self.lon = lon
        self.city = city
        self.country = country
        self.source = source


def get_shard_urls(
    hf_repo: str = "sdan/geospot-unified",
    max_shards: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Get shard URLs from manifest or by listing repo files."""
    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"

    # Try manifest first (fast path)
    manifest_url = f"{base_url}/shards.txt"
    tar_files = _try_load_manifest(manifest_url)

    if tar_files is None:
        # Fallback: list all files (slow)
        logger.info(f"No manifest found, listing files from {hf_repo}...")
        all_files = list_repo_files(hf_repo, repo_type="dataset")
        tar_files = [f for f in all_files if f.endswith(".tar")]

    logger.info(f"Found {len(tar_files)} shards")

    random.seed(seed)
    random.shuffle(tar_files)

    if max_shards:
        tar_files = tar_files[:max_shards]

    urls = [f"{base_url}/{path}" for path in tar_files]
    logger.info(f"Using {len(urls)} shards")
    return urls


def _try_load_manifest(url: str) -> list[str] | None:
    """Try to load shard paths from manifest file."""
    import requests
    try:
        resp = requests.get(url, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        paths = [line.strip() for line in resp.text.splitlines() if line.strip()]
        logger.info(f"Loaded {len(paths)} shards from manifest")
        return paths
    except Exception:
        return None


def iterate_samples(
    hf_repo: str = "sdan/geospot-unified",
    max_shards: int | None = None,
    seed: int = 0,
    shuffle_buffer: int = 1000,
    max_image_size: int = 512,
) -> Iterator[GeoSample]:
    """Stream GeoSamples using webdataset for efficient streaming."""
    urls = get_shard_urls(hf_repo, max_shards=max_shards, seed=seed)

    def warn_and_continue(exn):
        logger.debug(f"Skipping sample: {exn}")
        return True

    # webdataset streams directly without resolving all files
    dataset = (
        wds.WebDataset(urls, shardshuffle=100, handler=warn_and_continue)
        .shuffle(shuffle_buffer)
        .decode("pil", handler=warn_and_continue)
    )

    for sample in dataset:
        geo_sample = _parse_sample(sample, max_image_size)
        if geo_sample is not None:
            yield geo_sample


def _resize_and_crop(image: Image.Image, target_size: int, random_crop: bool = True) -> Image.Image:
    """Scale shortest side to target_size, then crop to square."""
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # Crop to square
    if random_crop:
        left = random.randint(0, new_w - target_size)
        top = random.randint(0, new_h - target_size)
    else:
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
    return image.crop((left, top, left + target_size, top + target_size))


def _parse_sample(sample: dict, max_image_size: int) -> GeoSample | None:
    """Parse a webdataset sample into a GeoSample."""
    try:
        image = sample.get("jpg") or sample.get("png")
        if image is None:
            return None

        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        # Resize shortest side + random crop for augmentation
        if max_image_size:
            image = _resize_and_crop(image, max_image_size, random_crop=True)

        json_data = sample.get("json", {})
        lat = float(json_data.get("lat", 0))
        lon = float(json_data.get("lon", 0))

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None

        return GeoSample(
            image=image,
            lat=lat,
            lon=lon,
            city=json_data.get("city"),
            country=json_data.get("country"),
            source=json_data.get("source"),
        )
    except Exception as e:
        logger.debug(f"Failed to parse sample: {e}")
        return None

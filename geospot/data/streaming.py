"""
Streaming data loading for geolocation datasets.

Supports:
- sdan/geomix (default, has train/val splits)
- sdan/geospot-unified (legacy)

Uses webdataset for efficient streaming without resolving all files upfront.
"""

import logging
import random
from typing import Iterator

import webdataset as wds
from huggingface_hub import list_repo_files
from PIL import Image

logger = logging.getLogger(__name__)

# Dataset configs: repo -> (manifest_name, base_path)
DATASET_CONFIGS = {
    "sdan/geomix": {
        "train": "train_shards.txt",
        "val": "val_shards.txt",
    },
    "sdan/geospot-unified": {
        "train": "shards.txt",
    },
}


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
    hf_repo: str = "sdan/geomix",
    split: str = "train",
    max_shards: int | None = None,
    seed: int = 0,
    local_path: str | None = None,
) -> list[str]:
    """Get shard URLs from manifest, local path, or by listing repo files.

    Args:
        local_path: If provided, read shards from this local directory instead of HuggingFace.
                    e.g., "/root/.cache/user_artifacts/geomix" for Baseten cache.
    """
    import glob as globmod

    # Local path mode: glob for .tar files directly
    if local_path:
        pattern = f"{local_path}/**/*.tar"
        tar_files = sorted(globmod.glob(pattern, recursive=True))
        if not tar_files:
            # Try non-recursive
            tar_files = sorted(globmod.glob(f"{local_path}/*.tar"))
        logger.info(f"Found {len(tar_files)} local shards in {local_path}")

        random.seed(seed)
        random.shuffle(tar_files)
        if max_shards:
            tar_files = tar_files[:max_shards]
        return tar_files

    # HuggingFace mode (original behavior)
    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"

    # Look up manifest name from config
    config = DATASET_CONFIGS.get(hf_repo, {})
    manifest_name = config.get(split)

    tar_files = None
    if manifest_name:
        manifest_url = f"{base_url}/{manifest_name}"
        tar_files = _try_load_manifest(manifest_url)

    if tar_files is None:
        # Fallback: list all files (slow)
        logger.info(f"No manifest found, listing files from {hf_repo}...")
        all_files = list_repo_files(hf_repo, repo_type="dataset")
        tar_files = [f for f in all_files if f.endswith(".tar")]

    logger.info(f"Found {len(tar_files)} shards for {hf_repo} ({split})")

    random.seed(seed)
    random.shuffle(tar_files)

    if max_shards:
        tar_files = tar_files[:max_shards]

    # Handle both full URLs and relative paths in manifest
    urls = []
    for path in tar_files:
        if path.startswith("http"):
            urls.append(path)
        else:
            urls.append(f"{base_url}/{path}")

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
    hf_repo: str = "sdan/geomix",
    split: str = "train",
    max_shards: int | None = None,
    seed: int = 0,
    shuffle_buffer: int = 1000,
    max_image_size: int = 512,
    local_path: str | None = None,
) -> Iterator[GeoSample]:
    """Stream GeoSamples using webdataset for efficient streaming.

    Args:
        local_path: If provided, read from local cache instead of HuggingFace.
                    e.g., "$BT_PROJECT_CACHE_DIR/geomix" for Baseten.
    """
    urls = get_shard_urls(hf_repo, split=split, max_shards=max_shards, seed=seed, local_path=local_path)

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

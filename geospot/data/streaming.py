"""
Streaming data loading for geolocation datasets.

Supports:
- osv5m/osv5m (5M streetview images, HuggingFace datasets format)
- sdan/geomix (webdataset format with train/val splits)
- sdan/geospot-unified (legacy webdataset)

Uses webdataset for tar-based datasets, HuggingFace datasets for others.
"""

import logging
import random
from typing import Iterator

import webdataset as wds
from huggingface_hub import list_repo_files
from PIL import Image

logger = logging.getLogger(__name__)

# Dataset configs: repo -> format info
# "webdataset" format uses .tar shards, "hf" uses HuggingFace datasets
DATASET_CONFIGS = {
    "osv5m/osv5m": {
        "format": "hf",
        "image_column": "image",
        "lat_column": "latitude",
        "lon_column": "longitude",
        "country_column": "country",
        "city_column": None,  # OSV-5M doesn't have city
    },
    "sdan/geomix": {
        "format": "webdataset",
        "train": "train_shards.txt",
        "val": "val_shards.txt",
    },
    "sdan/geospot-unified": {
        "format": "webdataset",
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
                    e.g., "/cache/osv5m" for Modal volume.
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
    hf_repo: str = "osv5m/osv5m",
    split: str = "train",
    max_shards: int | None = None,
    seed: int = 0,
    shuffle_buffer: int = 1000,
    max_image_size: int = 512,
    local_path: str | None = None,
) -> Iterator[GeoSample]:
    """Stream GeoSamples from various dataset formats.

    Auto-detects format based on hf_repo:
    - osv5m/osv5m: HuggingFace datasets (streaming)
    - sdan/geomix: webdataset (.tar shards)

    Args:
        local_path: If provided, read from local cache instead of HuggingFace.
                    e.g., "/cache/osv5m" for Modal.
    """
    config = DATASET_CONFIGS.get(hf_repo, {"format": "webdataset"})
    dataset_format = config.get("format", "webdataset")

    if dataset_format == "hf":
        yield from _iterate_hf_samples(
            hf_repo=hf_repo,
            split=split,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
            max_image_size=max_image_size,
            config=config,
        )
    else:
        yield from _iterate_webdataset_samples(
            hf_repo=hf_repo,
            split=split,
            max_shards=max_shards,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
            max_image_size=max_image_size,
            local_path=local_path,
        )


def _iterate_hf_samples(
    hf_repo: str,
    split: str,
    seed: int,
    shuffle_buffer: int,
    max_image_size: int,
    config: dict,
) -> Iterator[GeoSample]:
    """Stream samples from HuggingFace datasets (e.g., osv5m/osv5m)."""
    from datasets import load_dataset

    logger.info(
        "Loading %s (%s) via HuggingFace datasets streaming (streaming=True)...",
        hf_repo,
        split,
    )

    # Load with streaming for memory efficiency
    ds = load_dataset(hf_repo, split=split, streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    image_col = config.get("image_column", "image")
    lat_col = config.get("lat_column", "latitude")
    lon_col = config.get("lon_column", "longitude")
    country_col = config.get("country_column")
    city_col = config.get("city_column")

    for idx, sample in enumerate(ds):
        try:
            image = sample.get(image_col)
            if image is None:
                continue

            # Handle PIL image or bytes
            if isinstance(image, bytes):
                import io
                image = Image.open(io.BytesIO(image))
            elif not isinstance(image, Image.Image):
                continue

            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            if max_image_size:
                image = _resize_and_crop(image, max_image_size, random_crop=True)

            lat = float(sample.get(lat_col, 0))
            lon = float(sample.get(lon_col, 0))

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            if idx == 0:
                logger.info("Received first streamed sample from %s (%s).", hf_repo, split)

            yield GeoSample(
                image=image,
                lat=lat,
                lon=lon,
                city=sample.get(city_col) if city_col else None,
                country=sample.get(country_col) if country_col else None,
                source=hf_repo,
            )
        except Exception as e:
            logger.debug(f"Skipping sample: {e}")
            continue


def _iterate_webdataset_samples(
    hf_repo: str,
    split: str,
    max_shards: int | None,
    seed: int,
    shuffle_buffer: int,
    max_image_size: int,
    local_path: str | None,
) -> Iterator[GeoSample]:
    """Stream samples from webdataset (.tar shards)."""
    urls = get_shard_urls(hf_repo, split=split, max_shards=max_shards, seed=seed, local_path=local_path)

    def warn_and_continue(exn):
        logger.debug(f"Skipping sample: {exn}")
        return True

    # webdataset streams directly without resolving all files
    dataset = (
        wds.WebDataset(
            urls,
            shardshuffle=100,
            handler=warn_and_continue,
            empty_check=False,
        )
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
        image = (
            sample.get("jpg")
            or sample.get("jpeg")
            or sample.get("png")
            or sample.get("webp")
            or sample.get("image")
        )
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

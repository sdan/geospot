"""
Streaming data loading for geospot-unified dataset.
"""

import logging
from typing import Any, Iterator

import webdataset as wds
from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


def get_shard_urls(
    hf_repo: str = "sdan/geospot-unified",
    split: str = "train",
    max_shards: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Get shard URLs from HuggingFace manifest dataset."""
    logger.info(f"Loading manifest from {hf_repo} ({split})...")

    ds = load_dataset(hf_repo, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    # Take max_shards if specified, otherwise take all
    if max_shards:
        paths = [sample["text"] for sample in ds.take(max_shards)]
    else:
        paths = [sample["text"] for sample in ds]

    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    urls = [f"{base_url}/{path}" for path in paths]

    logger.info(f"Using {len(urls)} shards")
    return urls


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


def iterate_samples(
    urls: list[str],
    shuffle_buffer: int = 1000,
    max_image_size: int = 480,
) -> Iterator[GeoSample]:
    """Stream GeoSamples from webdataset shards."""

    def warn_and_continue(exn):
        logger.warning(f"Skipping sample: {exn}")
        return True

    dataset = (
        wds.WebDataset(urls, shardshuffle=1000, handler=warn_and_continue)
        .shuffle(shuffle_buffer)
        .decode("pil", handler=warn_and_continue)
    )

    for sample in dataset:
        geo_sample = _parse_sample(sample, max_image_size)
        if geo_sample is not None:
            yield geo_sample


def _parse_sample(sample: dict[str, Any], max_image_size: int) -> GeoSample | None:
    """Parse a webdataset sample into a GeoSample."""
    try:
        # Get image
        image = sample.get("jpg") or sample.get("png")
        if image is None:
            return None

        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        # Resize if needed
        if max_image_size and max(image.size) > max_image_size:
            ratio = max_image_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Get coordinates from JSON
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

"""
Dataset and builder for geolocation RL training.
"""

import io
import logging
import math
from functools import partial
from typing import Any, Iterator, Sequence, cast

import chz
from datasets import Dataset, load_dataset
from PIL import Image

from geospot.data import GeoSample, iterate_samples
from geospot.rl.types import RLDataset, RLDatasetBuilder, EnvGroupBuilder
from geospot.rl.geo_env import GeoEnv, GeoEnvConfig, GeoGroupBuilder
from geospot.rl.geo_reward import GeoLocation, GeoRewardConfig
from geospot.renderers import Renderer, get_renderer
from geospot.tokenizer_utils import get_tokenizer
from geospot.image_processing_utils import get_image_processor

logger = logging.getLogger(__name__)


class GeoDataset(RLDataset):
    """RL dataset for geo images from HuggingFace."""

    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        env_config: GeoEnvConfig | None = None,
        image_column: str = "image",
        lat_column: str = "lat",
        lon_column: str = "lon",
        city_column: str | None = "city",
        region_column: str | None = "region",
        country_column: str | None = "country",
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.env_config = env_config or GeoEnvConfig()

        self.image_column = image_column
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.city_column = city_column
        self.region_column = region_column
        self.country_column = country_column

    def _get_value(self, row: dict[str, Any], key: str | None) -> Any:
        """Get value from row, handling nested 'json' dict structure."""
        if key is None:
            return None
        if key in row:
            return row[key]
        # Nested in 'json' dict (e.g., geospot-vista9 format)
        if "json" in row and isinstance(row["json"], dict):
            if key in row["json"]:
                return row["json"][key]
        return None

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row)) is not None
        ]

    def _make_env_group_builder(self, row: dict[str, Any]) -> GeoGroupBuilder | None:
        try:
            image = self._get_value(row, self.image_column)
            if image is None:
                return None
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif not isinstance(image, Image.Image):
                logger.warning(f"Unexpected image type: {type(image)}")
                return None

            lat = float(self._get_value(row, self.lat_column))
            lon = float(self._get_value(row, self.lon_column))

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None

            city = self._get_value(row, self.city_column)
            region = self._get_value(row, self.region_column)
            country = self._get_value(row, self.country_column)

            ground_truth = GeoLocation(
                lat=lat, lon=lon, city=city, region=region, country=country
            )

            return GeoGroupBuilder(
                env_thunk=partial(
                    GeoEnv,
                    image=image,
                    ground_truth=ground_truth,
                    renderer=self.renderer,
                    config=self.env_config,
                ),
                num_envs=self.group_size,
            )

        except Exception as e:
            logger.warning(f"Failed to create env from row: {e}")
            return None


class StreamingGeoDataset(RLDataset):
    """Streaming RL dataset using webdataset for manifest-based datasets like geospot-unified."""

    def __init__(
        self,
        hf_repo: str,
        group_size: int,
        renderer: Renderer,
        env_config: GeoEnvConfig | None = None,
        max_shards: int | None = None,
        seed: int = 0,
        local_path: str | None = None,
    ):
        self.hf_repo = hf_repo
        self.group_size = group_size
        self.renderer = renderer
        self.env_config = env_config or GeoEnvConfig()
        self.max_shards = max_shards
        self.seed = seed
        self.local_path = local_path  # e.g., "/cache/osv5m" for Modal volume
        self._sample_iter: Iterator[GeoSample] | None = None

    def _get_sample_iter(self) -> Iterator[GeoSample]:
        if self._sample_iter is None:
            self._sample_iter = iterate_samples(
                hf_repo=self.hf_repo,
                max_shards=self.max_shards,
                seed=self.seed,
                local_path=self.local_path,
            )
        return self._sample_iter

    def reset(self, seed: int | None = None):
        """Reset the iterator with optional new seed."""
        if seed is not None:
            self.seed = seed
        self._sample_iter = None

    def __len__(self) -> int:
        return -1  # Streaming, unknown length

    def get_batch(self, batch_size: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of EnvGroupBuilders."""
        sample_iter = self._get_sample_iter()
        builders: list[GeoGroupBuilder] = []

        while len(builders) < batch_size:
            try:
                sample = next(sample_iter)
                ground_truth = GeoLocation(
                    lat=sample.lat,
                    lon=sample.lon,
                    city=sample.city,
                    country=sample.country,
                )
                builder = GeoGroupBuilder(
                    env_thunk=partial(
                        GeoEnv,
                        image=sample.image,
                        ground_truth=ground_truth,
                        renderer=self.renderer,
                        config=self.env_config,
                    ),
                    num_envs=self.group_size,
                    dataset_name=sample.source or "geospot",
                )
                builders.append(builder)
            except StopIteration:
                logger.info("Dataset exhausted, reshuffling...")
                self.reset(self.seed + 1)

        return builders


@chz.chz
class GeoDatasetBuilder(RLDatasetBuilder):
    """Builder for GeoDataset."""

    hf_repo: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int

    hf_split: str = "train"
    hf_test_split: str | None = "test"
    max_samples: int | None = None
    seed: int = 0

    image_column: str = "jpg"  # geospot-vista9 uses 'jpg'
    lat_column: str = "lat"
    lon_column: str = "lon"

    max_image_size: int = 480
    reward_config: GeoRewardConfig | None = None

    async def __call__(self) -> tuple[GeoDataset, GeoDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        env_config = GeoEnvConfig(
            max_image_size=self.max_image_size,
            reward_config=self.reward_config,
        )

        train_ds = self._load_split(self.hf_split)
        train_dataset = GeoDataset(
            ds=train_ds,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            env_config=env_config,
            image_column=self.image_column,
            lat_column=self.lat_column,
            lon_column=self.lon_column,
        )

        test_dataset = None
        if self.hf_test_split:
            try:
                test_ds = self._load_split(self.hf_test_split)
                test_dataset = GeoDataset(
                    ds=test_ds,
                    batch_size=self.batch_size,
                    group_size=1,
                    renderer=renderer,
                    env_config=env_config,
                    image_column=self.image_column,
                    lat_column=self.lat_column,
                    lon_column=self.lon_column,
                )
            except Exception as e:
                logger.warning(f"Could not load test split: {e}")

        return train_dataset, test_dataset

    def _load_split(self, split: str) -> Dataset:
        logger.info(f"Loading {self.hf_repo} split={split} (streaming)")
        ds = load_dataset(self.hf_repo, split=split, streaming=True)

        # Take max_samples and convert to regular dataset
        if self.max_samples:
            samples = list(ds.take(self.max_samples))
        else:
            samples = list(ds.take(10000))  # reasonable default

        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(samples)
        ds = ds.shuffle(seed=self.seed)
        logger.info(f"Loaded {len(ds):,} samples")
        return ds


@chz.chz
class OSV5MDatasetBuilder(GeoDatasetBuilder):
    """Pre-configured builder for OSV-5M dataset."""

    hf_repo: str = "osv5m/osv5m"
    image_column: str = "image"
    lat_column: str = "latitude"
    lon_column: str = "longitude"


@chz.chz
class StreamingGeoDatasetBuilder(RLDatasetBuilder):
    """Builder for StreamingGeoDataset (supports osv5m/osv5m, sdan/geomix, etc.)."""

    hf_repo: str = "osv5m/osv5m"
    group_size: int = 16
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    renderer_name: str = "qwen3_vl"

    max_shards: int | None = None
    seed: int = 0
    max_image_size: int = 480
    reward_config: GeoRewardConfig | None = None

    # Local cache path (e.g., Modal volume). If set, reads from local disk instead of HuggingFace.
    local_path: str | None = None

    async def __call__(self) -> tuple[StreamingGeoDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        env_config = GeoEnvConfig(
            max_image_size=self.max_image_size,
            reward_config=self.reward_config,
        )

        dataset = StreamingGeoDataset(
            hf_repo=self.hf_repo,
            group_size=self.group_size,
            renderer=renderer,
            env_config=env_config,
            max_shards=self.max_shards,
            seed=self.seed,
            local_path=self.local_path,
        )

        return dataset, None  # No test set for streaming


# -----------------------------------------------------------------------------
# Hierarchical (multi-turn) dataset builders
# -----------------------------------------------------------------------------

from geospot.rl.hierarchical_geo_env import (
    HierarchicalGeoEnv,
    HierarchicalGeoEnvConfig,
    HierarchicalGeoGroupBuilder,
)


class HierarchicalGeoDataset(RLDataset):
    """RL dataset for hierarchical multi-turn geo training."""

    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        env_config: HierarchicalGeoEnvConfig | None = None,
        image_column: str = "image",
        lat_column: str = "lat",
        lon_column: str = "lon",
        city_column: str | None = "city",
        region_column: str | None = "region",
        country_column: str | None = "country",
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.env_config = env_config or HierarchicalGeoEnvConfig()

        self.image_column = image_column
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.city_column = city_column
        self.region_column = region_column
        self.country_column = country_column

    def _get_value(self, row: dict[str, Any], key: str | None) -> Any:
        """Get value from row, handling nested 'json' dict structure."""
        if key is None:
            return None
        if key in row:
            return row[key]
        if "json" in row and isinstance(row["json"], dict):
            if key in row["json"]:
                return row["json"][key]
        return None

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row)) is not None
        ]

    def _make_env_group_builder(self, row: dict[str, Any]) -> HierarchicalGeoGroupBuilder | None:
        try:
            image = self._get_value(row, self.image_column)
            if image is None:
                return None
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif not isinstance(image, Image.Image):
                logger.warning(f"Unexpected image type: {type(image)}")
                return None

            lat = float(self._get_value(row, self.lat_column))
            lon = float(self._get_value(row, self.lon_column))

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None

            city = self._get_value(row, self.city_column)
            region = self._get_value(row, self.region_column)
            country = self._get_value(row, self.country_column)

            # Skip samples without country/region labels (needed for hierarchical)
            if not country:
                logger.debug("Skipping sample without country label")
                return None

            ground_truth = GeoLocation(
                lat=lat, lon=lon, city=city, region=region, country=country
            )

            return HierarchicalGeoGroupBuilder(
                env_thunk=partial(
                    HierarchicalGeoEnv,
                    image=image,
                    ground_truth=ground_truth,
                    renderer=self.renderer,
                    config=self.env_config,
                ),
                num_envs=self.group_size,
            )

        except Exception as e:
            logger.warning(f"Failed to create hierarchical env from row: {e}")
            return None


@chz.chz
class HierarchicalGeoDatasetBuilder(RLDatasetBuilder):
    """Builder for hierarchical multi-turn geo training."""

    hf_repo: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int

    hf_split: str = "train"
    hf_test_split: str | None = "test"
    max_samples: int | None = None
    seed: int = 0

    image_column: str = "jpg"
    lat_column: str = "lat"
    lon_column: str = "lon"
    city_column: str | None = "city"
    region_column: str | None = "region"
    country_column: str | None = "country"

    # Hierarchical-specific config
    max_image_size: int = 480
    teacher_forcing_prob: float = 1.0
    turns: list[str] = chz.field(default_factory=lambda: ["country", "region", "coords"])
    reward_config: GeoRewardConfig | None = None

    async def __call__(self) -> tuple[HierarchicalGeoDataset, HierarchicalGeoDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        env_config = HierarchicalGeoEnvConfig(
            turns=self.turns,
            teacher_forcing_prob=self.teacher_forcing_prob,
            max_image_size=self.max_image_size,
            reward_config=self.reward_config,
        )

        train_ds = self._load_split(self.hf_split)
        train_dataset = HierarchicalGeoDataset(
            ds=train_ds,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            env_config=env_config,
            image_column=self.image_column,
            lat_column=self.lat_column,
            lon_column=self.lon_column,
            city_column=self.city_column,
            region_column=self.region_column,
            country_column=self.country_column,
        )

        test_dataset = None
        if self.hf_test_split:
            try:
                test_ds = self._load_split(self.hf_test_split)
                # Test with no teacher forcing (realistic evaluation)
                test_env_config = HierarchicalGeoEnvConfig(
                    turns=self.turns,
                    teacher_forcing_prob=0.0,  # No hints during eval
                    max_image_size=self.max_image_size,
                    reward_config=self.reward_config,
                )
                test_dataset = HierarchicalGeoDataset(
                    ds=test_ds,
                    batch_size=self.batch_size,
                    group_size=1,  # No GRPO for eval
                    renderer=renderer,
                    env_config=test_env_config,
                    image_column=self.image_column,
                    lat_column=self.lat_column,
                    lon_column=self.lon_column,
                    city_column=self.city_column,
                    region_column=self.region_column,
                    country_column=self.country_column,
                )
            except Exception as e:
                logger.warning(f"Could not load test split: {e}")

        return train_dataset, test_dataset

    def _load_split(self, split: str) -> Dataset:
        logger.info(f"Loading {self.hf_repo} split={split} (streaming)")
        ds = load_dataset(self.hf_repo, split=split, streaming=True)

        if self.max_samples:
            samples = list(ds.take(self.max_samples))
        else:
            samples = list(ds.take(10000))

        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(samples)
        ds = ds.shuffle(seed=self.seed)
        logger.info(f"Loaded {len(ds):,} samples")
        return ds


# -----------------------------------------------------------------------------
# Geohash Curriculum (no labels needed, just lat/lon)
# -----------------------------------------------------------------------------

from geospot.rl.geohash_curriculum_env import (
    GeohashCurriculumEnv,
    GeohashCurriculumConfig,
    GeohashCurriculumGroupBuilder,
)


class GeohashCurriculumDataset(RLDataset):
    """RL dataset for geohash-based curriculum training. Works with any dataset that has lat/lon."""

    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        env_config: GeohashCurriculumConfig | None = None,
        image_column: str = "jpg",
        lat_column: str = "lat",
        lon_column: str = "lon",
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.env_config = env_config or GeohashCurriculumConfig()

        self.image_column = image_column
        self.lat_column = lat_column
        self.lon_column = lon_column

    def _get_value(self, row: dict[str, Any], key: str) -> Any:
        """Get value from row, handling nested 'json' dict structure."""
        if key in row:
            return row[key]
        if "json" in row and isinstance(row["json"], dict):
            if key in row["json"]:
                return row["json"][key]
        return None

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row)) is not None
        ]

    def _make_env_group_builder(self, row: dict[str, Any]) -> GeohashCurriculumGroupBuilder | None:
        try:
            image = self._get_value(row, self.image_column)
            if image is None:
                return None
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif not isinstance(image, Image.Image):
                logger.warning(f"Unexpected image type: {type(image)}")
                return None

            lat = float(self._get_value(row, self.lat_column))
            lon = float(self._get_value(row, self.lon_column))

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None

            # Only need lat/lon for geohash curriculum
            ground_truth = GeoLocation(lat=lat, lon=lon)

            return GeohashCurriculumGroupBuilder(
                env_thunk=partial(
                    GeohashCurriculumEnv,
                    image=image,
                    ground_truth=ground_truth,
                    renderer=self.renderer,
                    config=self.env_config,
                ),
                num_envs=self.group_size,
            )

        except Exception as e:
            logger.warning(f"Failed to create geohash curriculum env from row: {e}")
            return None


@chz.chz
class GeohashCurriculumDatasetBuilder(RLDatasetBuilder):
    """Builder for geohash-based curriculum training. Works with osv5m or any lat/lon dataset."""

    hf_repo: str = "osv5m/osv5m"
    batch_size: int = 32
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    renderer_name: str = "qwen3_vl"
    group_size: int = 8

    hf_split: str = "train"
    max_samples: int | None = None
    seed: int = 0

    image_column: str = "jpg"
    lat_column: str = "lat"
    lon_column: str = "lon"

    max_image_size: int = 480
    teacher_forcing_prob: float = 0.5  # Balanced: 50% ground truth hints, 50% own predictions
    improvement_bonus: float = 0.1

    async def __call__(self) -> tuple[GeohashCurriculumDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        env_config = GeohashCurriculumConfig(
            teacher_forcing_prob=self.teacher_forcing_prob,
            improvement_bonus=self.improvement_bonus,
            max_image_size=self.max_image_size,
        )

        train_ds = self._load_split(self.hf_split)
        train_dataset = GeohashCurriculumDataset(
            ds=train_ds,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            env_config=env_config,
            image_column=self.image_column,
            lat_column=self.lat_column,
            lon_column=self.lon_column,
        )

        return train_dataset, None

    def _load_split(self, split: str) -> Dataset:
        logger.info(f"Loading {self.hf_repo} split={split} (streaming)")
        ds = load_dataset(self.hf_repo, split=split, streaming=True)

        if self.max_samples:
            samples = list(ds.take(self.max_samples))
        else:
            samples = list(ds.take(10000))

        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(samples)
        ds = ds.shuffle(seed=self.seed)
        logger.info(f"Loaded {len(ds):,} samples")
        return ds
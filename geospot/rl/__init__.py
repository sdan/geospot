"""RL module for geospot VLM training."""

from geospot.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Transition,
    Trajectory,
    TrajectoryGroup,
    Action,
    Observation,
    Metrics,
)
from geospot.rl.geo_env import GeoEnv, GeoGroupBuilder, GeoEnvConfig
from geospot.rl.geo_dataset import GeoDataset, GeoDatasetBuilder, OSV5MDatasetBuilder
from geospot.rl.telescoping_geo_env import (
    TelescopingGeoEnv,
    TelescopingGeoEnvConfig,
    TelescopingGeoGroupBuilder,
)
from geospot.rl.geo_reward import (
    GeoLocation,
    GeoRewardConfig,
    GeoRewardResult,
    ParsedGeoResponse,
    ReverseGeocoder,
    geo_location_from_coords,
    haversine_km,
    compute_geo_reward,
    parse_geo_response,
    distance_bucket,
    geoguessr_score,
)

__all__ = [
    # Core types
    "Env",
    "EnvGroupBuilder",
    "RLDataset",
    "RLDatasetBuilder",
    "StepResult",
    "Transition",
    "Trajectory",
    "TrajectoryGroup",
    "Action",
    "Observation",
    "Metrics",
    # Geo env
    "GeoEnv",
    "GeoGroupBuilder",
    "GeoEnvConfig",
    # Geo dataset
    "GeoDataset",
    "GeoDatasetBuilder",
    "OSV5MDatasetBuilder",
    # Telescoping env
    "TelescopingGeoEnv",
    "TelescopingGeoEnvConfig",
    "TelescopingGeoGroupBuilder",
    # Geo reward
    "GeoLocation",
    "GeoRewardConfig",
    "GeoRewardResult",
    "ParsedGeoResponse",
    "ReverseGeocoder",
    "geo_location_from_coords",
    "haversine_km",
    "compute_geo_reward",
    "parse_geo_response",
    "distance_bucket",
    "geoguessr_score",
]

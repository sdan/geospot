"""Geospot VLM - GRPO training for visual geolocation."""

from geospot.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    Transition,
    Trajectory,
    TrajectoryGroup,
    TokensWithLogprobs,
    Action,
    Observation,
    StopCondition,
    Metrics,
)
from geospot.envs import (
    # Single-turn (simple)
    SingleTurnGeoEnv,
    SingleTurnGeoEnvConfig,
    SingleTurnGeoGroupBuilder,
    # Multi-turn (dense rewards)
    MultiTurnGeoEnv,
    MultiTurnGeoEnvConfig,
    MultiTurnGeoGroupBuilder,
    # Utilities
    GeoLocation,
    ParsedGeoResponse,
    parse_geo_response,
)

__all__ = [
    # Types
    "Env", "EnvGroupBuilder", "StepResult", "Transition", "Trajectory",
    "TrajectoryGroup", "TokensWithLogprobs", "Action", "Observation",
    "StopCondition", "Metrics",
    # Single-turn env
    "SingleTurnGeoEnv", "SingleTurnGeoEnvConfig", "SingleTurnGeoGroupBuilder",
    # Multi-turn env
    "MultiTurnGeoEnv", "MultiTurnGeoEnvConfig", "MultiTurnGeoGroupBuilder",
    # Utilities
    "GeoLocation", "ParsedGeoResponse", "parse_geo_response",
]

__version__ = "0.1.0"

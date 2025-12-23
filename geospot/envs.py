"""
Geolocation environments for GRPO training.

Two variants:
- SingleTurnGeoEnv: One image → one guess → one reward (simple)
- MultiTurnGeoEnv: Country → Region → City → Coords (dense rewards)
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

import tinker
from PIL import Image

from geospot.renderers import ImagePart, Message, Renderer, TextPart, ensure_text
from geospot.types import Action, Env, EnvGroupBuilder, Metrics, Observation, StepResult, StopCondition, Trajectory

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# =============================================================================
# Geo utilities
# =============================================================================


class GeoLocation(NamedTuple):
    """Geographic coordinates with metadata from OSV-5M."""
    lat: float
    lon: float
    country: str | None = None
    region: str | None = None
    sub_region: str | None = None
    city: str | None = None


class ParsedGeoResponse(NamedTuple):
    """Result of parsing model output."""
    location: GeoLocation | None
    country: str | None  # For multi-turn
    region: str | None   # For multi-turn
    city: str | None     # For multi-turn
    format_valid: bool
    raw_text: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def geoguessr_score(distance_km: float) -> int:
    """GeoGuessr-style score (0-5000)."""
    if distance_km < 0.05:
        return 5000
    return max(0, int(5000 * math.exp(-distance_km / 2000)))


def distance_bucket(distance_km: float) -> str:
    """For stratified eval metrics."""
    if distance_km < 1:
        return "<1km"
    elif distance_km < 25:
        return "1-25km"
    elif distance_km < 200:
        return "25-200km"
    elif distance_km < 750:
        return "200-750km"
    elif distance_km < 2500:
        return "750-2500km"
    return ">2500km"


def parse_geo_response(response: str) -> ParsedGeoResponse:
    """
    Parse model response for coordinates and/or text labels.

    Accepts coords:
        Latitude: <degrees>
        Longitude: <degrees>
    Or:
        <lat>, <lon>

    Accepts labels:
        Country: <name>
        Region: <name>
        City: <name>
    """
    response = response.strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    location = None
    country = None
    region = None
    city = None

    # Parse coordinates
    lat_match = re.search(r"Latitude:\s*(-?\d+\.?\d*)", response)
    lon_match = re.search(r"Longitude:\s*(-?\d+\.?\d*)", response)

    if lat_match and lon_match:
        try:
            lat, lon = float(lat_match.group(1)), float(lon_match.group(1))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                location = GeoLocation(lat, lon)
        except ValueError:
            pass

    # Try bare coords: "lat, lon"
    if location is None:
        bare_match = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", response)
        if bare_match:
            try:
                lat, lon = float(bare_match.group(1)), float(bare_match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    location = GeoLocation(lat, lon)
            except ValueError:
                pass

    # Parse text labels (for multi-turn)
    country_match = re.search(r"Country:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    region_match = re.search(r"Region:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    city_match = re.search(r"City:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

    if country_match:
        country = country_match.group(1).strip()
    if region_match:
        region = region_match.group(1).strip()
    if city_match:
        city = city_match.group(1).strip()

    format_valid = location is not None or country is not None
    return ParsedGeoResponse(location, country, region, city, format_valid, response)


def normalize_label(s: str | None) -> str:
    """Normalize for fuzzy matching."""
    if s is None:
        return ""
    return s.lower().strip()


def labels_match(pred: str | None, gt: str | None) -> bool:
    """Fuzzy match for country/region/city."""
    if pred is None or gt is None:
        return False
    pred_norm = normalize_label(pred)
    gt_norm = normalize_label(gt)
    # Exact or substring match
    return pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm


# =============================================================================
# Prompts
# =============================================================================

SINGLE_TURN_PROMPT = """Where is this? Reply with coordinates.

Latitude: <degrees>
Longitude: <degrees>"""

MULTI_TURN_PROMPTS = {
    "country": """Where is this? What country is this in?

Country: <name>""",

    "region": """You said {prev_country}. Now what region/state is this in?

Region: <name>""",

    "city": """You said {prev_region}. Now what city is this near?

City: <name>""",

    "coords": """You said {prev_city}. Now give your final coordinates.

Latitude: <degrees>
Longitude: <degrees>""",
}


# =============================================================================
# Single-Turn Environment (Simple)
# =============================================================================


@dataclass
class SingleTurnGeoEnvConfig:
    """Configuration for single-turn environment."""
    max_image_size: int = 512
    format_penalty: float = 0.1


class SingleTurnGeoEnv(Env):
    """
    Single-turn geolocation: image → coords → reward.

    Reward = geoguessr_score(distance) / 5000
    """

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoLocation,
        renderer: Renderer,
        config: SingleTurnGeoEnvConfig | None = None,
    ):
        self.image = image
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.config = config or SingleTurnGeoEnvConfig()

        # Resize if needed
        if self.config.max_image_size:
            w, h = self.image.size
            if max(w, h) > self.config.max_image_size:
                scale = self.config.max_image_size / max(w, h)
                self.image = self.image.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )

        self.messages: list[Message] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self.messages = [Message(role="user", content=[
            ImagePart(type="image", image=self.image),
            TextPart(type="text", text=SINGLE_TURN_PROMPT),
        ])]
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        msg, _ = self.renderer.parse_response(action)
        text = ensure_text(msg["content"])
        parsed = parse_geo_response(text)

        if parsed.location is None:
            return StepResult(
                reward=-self.config.format_penalty,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"format_error": 1.0},
            )

        lat, lon = parsed.location.lat, parsed.location.lon
        dist = haversine_km(lat, lon, self.ground_truth.lat, self.ground_truth.lon)
        score = geoguessr_score(dist) / 5000.0

        metrics: Metrics = {
            "distance_km": dist,
            "distance_bucket": distance_bucket(dist),
            "score": score,
        }

        return StepResult(
            reward=score,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


# =============================================================================
# Multi-Turn Environment (Dense Rewards)
# =============================================================================


@dataclass
class MultiTurnGeoEnvConfig:
    """Configuration for multi-turn environment."""
    max_image_size: int = 512
    format_penalty: float = 0.1
    # Reward weights for each turn
    country_reward: float = 0.2
    region_reward: float = 0.2
    city_reward: float = 0.2
    # Remaining 0.4 comes from final coord score


class MultiTurnGeoEnv(Env):
    """
    Multi-turn geolocation using OSV-5M labels.

    Turn 1: Country → +0.2 if correct
    Turn 2: Region  → +0.2 if correct
    Turn 3: City    → +0.2 if correct
    Turn 4: Coords  → geoguessr_score * 0.4

    Total possible reward: 1.0
    """

    TURNS = ["country", "region", "city", "coords"]

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoLocation,
        renderer: Renderer,
        config: MultiTurnGeoEnvConfig | None = None,
    ):
        self.image = image
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.config = config or MultiTurnGeoEnvConfig()

        # Resize if needed
        if self.config.max_image_size:
            w, h = self.image.size
            if max(w, h) > self.config.max_image_size:
                scale = self.config.max_image_size / max(w, h)
                self.image = self.image.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )

        self.turn_idx = 0
        self.prev_answers: dict[str, str] = {}
        self.messages: list[Message] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _get_prompt(self, turn: str) -> str:
        template = MULTI_TURN_PROMPTS[turn]
        return template.format(
            prev_country=self.prev_answers.get("country", ""),
            prev_region=self.prev_answers.get("region", ""),
            prev_city=self.prev_answers.get("city", ""),
        )

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prompt = self._get_prompt("country")
        self.messages = [Message(role="user", content=[
            ImagePart(type="image", image=self.image),
            TextPart(type="text", text=prompt),
        ])]
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        msg, _ = self.renderer.parse_response(action)
        text = ensure_text(msg["content"])
        parsed = parse_geo_response(text)

        turn = self.TURNS[self.turn_idx]
        metrics: Metrics = {"turn": self.turn_idx, "turn_name": turn}

        # Calculate reward based on turn type
        if turn == "country":
            correct = labels_match(parsed.country, self.ground_truth.country)
            reward = self.config.country_reward if correct else 0.0
            self.prev_answers["country"] = parsed.country or "unknown"
            metrics["country_correct"] = int(correct)

        elif turn == "region":
            correct = labels_match(parsed.region, self.ground_truth.region)
            reward = self.config.region_reward if correct else 0.0
            self.prev_answers["region"] = parsed.region or "unknown"
            metrics["region_correct"] = int(correct)

        elif turn == "city":
            correct = labels_match(parsed.city, self.ground_truth.city)
            reward = self.config.city_reward if correct else 0.0
            self.prev_answers["city"] = parsed.city or "unknown"
            metrics["city_correct"] = int(correct)

        elif turn == "coords":
            if parsed.location is None:
                return StepResult(
                    reward=-self.config.format_penalty,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={**metrics, "format_error": 1.0},
                )

            lat, lon = parsed.location.lat, parsed.location.lon
            dist = haversine_km(lat, lon, self.ground_truth.lat, self.ground_truth.lon)
            # Scale to 0.4 max (country+region+city take 0.6)
            reward = (geoguessr_score(dist) / 5000.0) * 0.4
            metrics["distance_km"] = dist
            metrics["distance_bucket"] = distance_bucket(dist)
            metrics["coord_score"] = reward

        self.turn_idx += 1
        done = self.turn_idx >= len(self.TURNS)

        if done:
            return StepResult(reward, True, tinker.ModelInput.empty(), self.stop_condition, metrics)

        # Next turn
        self.messages.append(Message(role="assistant", content=parsed.raw_text))
        next_prompt = self._get_prompt(self.TURNS[self.turn_idx])
        self.messages.append(Message(role="user", content=next_prompt))

        return StepResult(
            reward=reward,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.messages),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


# =============================================================================
# Group Builders (for GRPO)
# =============================================================================


@dataclass(frozen=True)
class SingleTurnGeoGroupBuilder(EnvGroupBuilder):
    """Builds N copies of SingleTurnGeoEnv for GRPO."""
    env_thunk: Callable[[], SingleTurnGeoEnv]
    num_envs: int
    dataset_name: str = "geospot"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectories: list[Trajectory], envs: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectories]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


@dataclass(frozen=True)
class MultiTurnGeoGroupBuilder(EnvGroupBuilder):
    """Builds N copies of MultiTurnGeoEnv for GRPO."""
    env_thunk: Callable[[], MultiTurnGeoEnv]
    num_envs: int
    dataset_name: str = "geospot"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectories: list[Trajectory], envs: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectories]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]

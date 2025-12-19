"""
Hierarchical multi-turn geolocation RL environment.

Multi-turn: image -> country -> region -> coords, with optional teacher forcing.

This decomposes the hard geo prediction task into easier subtasks:
1. Country prediction (coarse)
2. Region/state prediction (medium)
3. Coordinate prediction (fine)

Each turn gets its own reward signal, enabling better credit assignment.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Literal, Sequence

import tinker
from PIL import Image

from geospot.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from geospot.rl.geo_reward import (
    GeoLocation,
    GeoRewardConfig,
    ReverseGeocoder,
    compute_geo_reward,
    parse_geo_response,
    distance_bucket,
    geohash_encode,
    common_prefix_len,
)
from geospot.renderers import ImagePart, Message, Renderer, TextPart, ensure_text
from geospot.completers import StopCondition

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Prompts for each turn
# -----------------------------------------------------------------------------

COUNTRY_PROMPT = """What country is this location in?

Reply with just the country name, nothing else."""

REGION_PROMPT_TEMPLATE = """This location is in {country}.

What region, state, or province is this? Reply with just the region name, nothing else."""

COORDS_PROMPT_TEMPLATE = """This location is in {region}, {country}.

What are the exact coordinates?

Latitude: <degrees>
Longitude: <degrees>"""

# Alternative: single-turn with thinking
HIERARCHICAL_SINGLE_TURN_PROMPT = """Where is this location?

Think step by step:
1. First identify the country
2. Then narrow down to the region/state
3. Finally estimate the coordinates

<country>country name</country>
<region>region name</region>
Latitude: <degrees>
Longitude: <degrees>"""


# -----------------------------------------------------------------------------
# Text normalization for matching
# -----------------------------------------------------------------------------

def normalize_text(s: str | None) -> str:
    """Normalize text for fuzzy matching."""
    if s is None:
        return ""
    s = s.lower().strip()
    # Remove common prefixes
    for prefix in ["the ", "republic of ", "state of ", "province of "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Remove punctuation and extra whitespace
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def texts_match(pred: str | None, truth: str | None) -> bool:
    """Check if two location texts match (fuzzy)."""
    pred_norm = normalize_text(pred)
    truth_norm = normalize_text(truth)
    if not pred_norm or not truth_norm:
        return False
    # Exact match
    if pred_norm == truth_norm:
        return True
    # One contains the other (handles "California" vs "California, USA")
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return True
    return False


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class HierarchicalGeoEnvConfig:
    """Configuration for hierarchical geo environment."""

    # Turn structure
    turns: list[Literal["country", "region", "coords"]] = field(
        default_factory=lambda: ["country", "region", "coords"]
    )

    # Teacher forcing: probability of giving ground truth hint for next turn
    # Set to 1.0 for full teacher forcing, 0.0 for autoregressive (use model's own predictions)
    teacher_forcing_prob: float = 1.0

    # Reward weights per turn (should sum to ~1.0)
    country_reward_weight: float = 0.2
    region_reward_weight: float = 0.3
    coords_reward_weight: float = 0.5

    # Partial credit for wrong answer but right "direction"
    # e.g., wrong country but same continent gets partial credit
    partial_credit: bool = True

    # Underlying coord reward config
    reward_config: GeoRewardConfig | None = None

    # Image settings
    max_image_size: int = 512

    # Format penalty
    format_coef: float = 0.1


# -----------------------------------------------------------------------------
# Hierarchical Environment
# -----------------------------------------------------------------------------

class HierarchicalGeoEnv(Env):
    """
    Multi-turn geo environment with hierarchical decomposition.

    Turn 1: "What country?" -> reward based on country match
    Turn 2: "What region in {country}?" -> reward based on region match
    Turn 3: "Coordinates in {region}, {country}?" -> reward based on distance

    Teacher forcing: Between turns, can give ground truth (easier) or model's
    own prediction (harder, more realistic).
    """

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoLocation,
        renderer: Renderer,
        config: HierarchicalGeoEnvConfig | None = None,
    ):
        self.image = image
        self.renderer = renderer
        self.config = config or HierarchicalGeoEnvConfig()

        # Auto-populate country/region via reverse geocoding if missing
        if ground_truth.country is None or ground_truth.region is None:
            geocoder = ReverseGeocoder()
            info = geocoder(ground_truth.lat, ground_truth.lon)
            self.ground_truth = GeoLocation(
                lat=ground_truth.lat,
                lon=ground_truth.lon,
                city=ground_truth.city or info.get("city"),
                region=ground_truth.region or info.get("region"),
                country=ground_truth.country or info.get("country"),
            )
        else:
            self.ground_truth = ground_truth

        if self.config.max_image_size:
            self.image = self._resize_image(self.image, self.config.max_image_size)

        # Turn tracking
        self.current_turn_idx = 0
        self.predictions: dict[str, str] = {}
        self.turn_rewards: dict[str, float] = {}
        self.messages: list[Message] = []

        # For geohash-based partial credit
        self.gt_geohash = geohash_encode(self.ground_truth.lat, self.ground_truth.lon, precision=5)

    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        w, h = image.size
        if max(w, h) <= max_size:
            return image
        if w > h:
            return image.resize((max_size, int(h * max_size / w)), Image.Resampling.LANCZOS)
        return image.resize((int(w * max_size / h), max_size), Image.Resampling.LANCZOS)

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @property
    def current_turn(self) -> str:
        if self.current_turn_idx < len(self.config.turns):
            return self.config.turns[self.current_turn_idx]
        return "done"

    @property
    def is_final_turn(self) -> bool:
        return self.current_turn_idx >= len(self.config.turns) - 1

    def _use_teacher_forcing(self) -> bool:
        """Decide whether to use ground truth or model prediction for hints."""
        return random.random() < self.config.teacher_forcing_prob

    def _get_hint(self, field: str) -> str:
        """Get hint for a field (ground truth or model prediction)."""
        if self._use_teacher_forcing():
            # Use ground truth
            if field == "country":
                return self.ground_truth.country or "Unknown"
            elif field == "region":
                return self.ground_truth.region or "Unknown"
        else:
            # Use model's own prediction
            return self.predictions.get(field, "Unknown")
        return "Unknown"

    def _get_prompt_for_turn(self, turn: str) -> str:
        """Build prompt for the given turn."""
        if turn == "country":
            return COUNTRY_PROMPT
        elif turn == "region":
            country = self._get_hint("country")
            return REGION_PROMPT_TEMPLATE.format(country=country)
        elif turn == "coords":
            country = self._get_hint("country")
            region = self._get_hint("region")
            return COORDS_PROMPT_TEMPLATE.format(country=country, region=region)
        else:
            raise ValueError(f"Unknown turn: {turn}")

    def _compute_turn_reward(self, turn: str, response_text: str) -> tuple[float, Metrics]:
        """Compute reward for a specific turn."""
        metrics: Metrics = {}

        if turn == "country":
            pred_country = response_text.strip()
            self.predictions["country"] = pred_country

            correct = texts_match(pred_country, self.ground_truth.country)
            reward = 1.0 if correct else 0.0

            metrics["country_correct"] = float(correct)
            metrics["country_pred"] = pred_country[:50]  # Truncate for logging

        elif turn == "region":
            pred_region = response_text.strip()
            self.predictions["region"] = pred_region

            correct = texts_match(pred_region, self.ground_truth.region)
            reward = 1.0 if correct else 0.0

            # Partial credit: if country was wrong, don't penalize region too harshly
            if self.config.partial_credit:
                country_was_correct = self.turn_rewards.get("country", 0) > 0.5
                if not country_was_correct and not correct:
                    reward = 0.25  # Small partial credit for trying

            metrics["region_correct"] = float(correct)
            metrics["region_pred"] = pred_region[:50]

        elif turn == "coords":
            parsed = parse_geo_response(response_text)
            reward_result = compute_geo_reward(
                prediction=parsed,
                ground_truth=self.ground_truth,
                config=self.config.reward_config,
            )
            reward = reward_result.total_reward

            # Add detailed coord metrics
            metrics.update(reward_result.to_metrics())
            if reward_result.distance_km is not None:
                metrics["distance_bucket"] = distance_bucket(reward_result.distance_km)

                # Geohash prefix match for interpretability
                if parsed.location:
                    pred_hash = geohash_encode(parsed.location.lat, parsed.location.lon, precision=5)
                    prefix_len = common_prefix_len(pred_hash, self.gt_geohash)
                    metrics["geohash_prefix_len"] = float(prefix_len)
        else:
            reward = 0.0

        self.turn_rewards[turn] = reward
        return reward, metrics

    def _get_reward_weight(self, turn: str) -> float:
        """Get reward weight for a turn."""
        if turn == "country":
            return self.config.country_reward_weight
        elif turn == "region":
            return self.config.region_reward_weight
        elif turn == "coords":
            return self.config.coords_reward_weight
        return 0.0

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Build initial observation with image and first prompt."""
        first_turn = self.config.turns[0]
        prompt = self._get_prompt_for_turn(first_turn)

        content = [
            ImagePart(type="image", image=self.image),
            TextPart(type="text", text=prompt),
        ]
        self.messages = [Message(role="user", content=content)]

        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Process model response and advance to next turn or finish."""
        message, parse_success = self.renderer.parse_response(action)
        response_text = ensure_text(message["content"])

        turn = self.current_turn

        # Compute reward for this turn
        turn_reward, turn_metrics = self._compute_turn_reward(turn, response_text)

        # Weight the reward
        weight = self._get_reward_weight(turn)
        weighted_reward = weight * turn_reward

        # Format penalty for non-final turns if unparseable
        if not parse_success and turn != "coords":
            weighted_reward -= self.config.format_coef

        # Add turn info to metrics
        turn_metrics["turn"] = self.current_turn_idx
        turn_metrics["turn_name"] = turn
        turn_metrics[f"reward/{turn}"] = turn_reward
        turn_metrics[f"reward/{turn}_weighted"] = weighted_reward

        # Advance turn
        self.current_turn_idx += 1
        episode_done = self.current_turn_idx >= len(self.config.turns)

        if episode_done:
            # Final metrics: total reward across all turns
            total_reward = sum(
                self._get_reward_weight(t) * self.turn_rewards.get(t, 0)
                for t in self.config.turns
            )
            turn_metrics["reward/total"] = total_reward
            turn_metrics["teacher_forcing_prob"] = self.config.teacher_forcing_prob

            next_obs = tinker.ModelInput.empty()
        else:
            # Build next turn observation
            self.messages.append(Message(role="assistant", content=response_text))

            next_turn = self.current_turn
            next_prompt = self._get_prompt_for_turn(next_turn)
            self.messages.append(Message(role="user", content=next_prompt))

            next_obs = self.renderer.build_generation_prompt(self.messages)

        return StepResult(
            reward=weighted_reward,
            episode_done=episode_done,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics=turn_metrics,
        )


# -----------------------------------------------------------------------------
# Group Builder (for GRPO)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchicalGeoGroupBuilder(EnvGroupBuilder):
    """Builder for groups of HierarchicalGeoEnv instances."""

    env_thunk: Callable[[], HierarchicalGeoEnv]
    num_envs: int
    dataset_name: str = "geospot-hierarchical"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """
        Compute group-level rewards (e.g., for GRPO baseline subtraction).

        For hierarchical env, we could add:
        - Bonus for being best in group at each level
        - Contrastive reward between trajectories
        """
        # For now, no group-level rewards (individual rewards sufficient)
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------

def create_hierarchical_env(
    image: Image.Image,
    ground_truth: GeoLocation,
    renderer: Renderer,
    teacher_forcing_prob: float = 1.0,
    turns: list[str] | None = None,
    reward_config: GeoRewardConfig | None = None,
) -> HierarchicalGeoEnv:
    """Factory function to create a hierarchical geo environment."""
    config = HierarchicalGeoEnvConfig(
        turns=turns or ["country", "region", "coords"],
        teacher_forcing_prob=teacher_forcing_prob,
        reward_config=reward_config,
    )
    return HierarchicalGeoEnv(
        image=image,
        ground_truth=ground_truth,
        renderer=renderer,
        config=config,
    )

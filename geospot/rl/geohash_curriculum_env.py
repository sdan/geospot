"""
Geohash-based curriculum geo environment.

Multi-turn curriculum using geohash precision levels instead of text labels.
NO country/region labels needed - just lat/lon.

Turns:
1. Coarse prediction → reward if geohash[0:2] matches (~600km)
2. Medium prediction → reward if geohash[0:4] matches (~40km)
3. Fine prediction → reward based on geohash[0:6] match (+ distance metrics)

This is cleaner than text-based hierarchy because:
- No need for reverse geocoding
- No text matching ambiguity ("United States" vs "USA")
- Purely spatial, mathematically precise
"""

import logging
import random
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Sequence

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
    compute_geo_reward,
    parse_geo_response,
    distance_bucket,
    geohash_encode,
    common_prefix_len,
    haversine_km,
    geoguessr_score,
)
from geospot.renderers import ImagePart, Message, Renderer, TextPart, ensure_text
from geospot.completers import StopCondition

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Prompts for each precision level
# -----------------------------------------------------------------------------

COARSE_PROMPT = """Where is this location? Give an approximate estimate.

Think about what continent and general region this might be.

Latitude: <degrees>
Longitude: <degrees>"""

MEDIUM_PROMPT_TEMPLATE = """Your initial guess was approximately ({prev_lat:.1f}, {prev_lon:.1f}).

Now refine your estimate. Consider country and region-level features.

Latitude: <degrees>
Longitude: <degrees>"""

FINE_PROMPT_TEMPLATE = """Your refined guess was ({prev_lat:.2f}, {prev_lon:.2f}).

Now give your final, most precise coordinates. Look for city-level and local features.

Latitude: <degrees>
Longitude: <degrees>"""


# -----------------------------------------------------------------------------
# Precision levels mapped to geohash
# -----------------------------------------------------------------------------

@dataclass
class PrecisionLevel:
    """A precision level in the curriculum."""
    name: str
    geohash_chars: int  # Number of geohash chars to match
    approx_km: float    # Approximate precision in km
    reward_weight: float
    prompt_template: str


DEFAULT_LEVELS = [
    PrecisionLevel("coarse", 2, 600, 0.2, COARSE_PROMPT),
    PrecisionLevel("medium", 4, 40, 0.3, MEDIUM_PROMPT_TEMPLATE),
    PrecisionLevel("fine", 6, 1, 0.5, FINE_PROMPT_TEMPLATE),
]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class GeohashCurriculumConfig:
    """Configuration for geohash curriculum environment."""

    levels: list[PrecisionLevel] = field(default_factory=lambda: DEFAULT_LEVELS.copy())

    # Teacher forcing: show previous ground-truth coords instead of model's prediction
    teacher_forcing_prob: float = 0.0  # Default off since we use model's own refinement

    # Bonus for improving from previous guess
    improvement_bonus: float = 0.1

    # Underlying coord reward config for final level
    reward_config: GeoRewardConfig | None = None

    # Image settings
    max_image_size: int = 512


# -----------------------------------------------------------------------------
# Geohash Curriculum Environment
# -----------------------------------------------------------------------------

class GeohashCurriculumEnv(Env):
    """
    Multi-turn geo environment using geohash precision curriculum.

    Each turn asks for progressively more precise coordinates.
    Rewards based on geohash prefix matching at each level.
    """

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoLocation,
        renderer: Renderer,
        config: GeohashCurriculumConfig | None = None,
    ):
        self.image = image
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.config = config or GeohashCurriculumConfig()

        if self.config.max_image_size:
            self.image = self._resize_image(self.image, self.config.max_image_size)

        # Precompute ground truth geohash
        max_precision = max(level.geohash_chars for level in self.config.levels)
        self.gt_geohash = geohash_encode(ground_truth.lat, ground_truth.lon, precision=max_precision)

        # Turn tracking
        self.current_level_idx = 0
        self.predictions: list[tuple[float, float]] = []  # (lat, lon) per turn
        self.level_rewards: dict[str, float] = {}
        self.messages: list[Message] = []

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
    def current_level(self) -> PrecisionLevel | None:
        if self.current_level_idx < len(self.config.levels):
            return self.config.levels[self.current_level_idx]
        return None

    @property
    def is_final_level(self) -> bool:
        return self.current_level_idx >= len(self.config.levels) - 1

    def _get_previous_coords(self) -> tuple[float, float] | None:
        """Get coords to show in prompt (teacher forcing or model's prediction)."""
        if not self.predictions:
            return None

        if random.random() < self.config.teacher_forcing_prob:
            # Teacher forcing: use ground truth
            return (self.ground_truth.lat, self.ground_truth.lon)
        else:
            # Use model's last prediction
            return self.predictions[-1]

    def _get_prompt_for_level(self, level: PrecisionLevel) -> str:
        """Build prompt for the given level."""
        prev = self._get_previous_coords()

        if prev is None:
            # First turn
            return level.prompt_template
        else:
            # Subsequent turns - format with previous coords
            try:
                return level.prompt_template.format(prev_lat=prev[0], prev_lon=prev[1])
            except KeyError:
                # Template doesn't have placeholders
                return level.prompt_template

    def _compute_level_reward(self, level: PrecisionLevel, pred_lat: float, pred_lon: float) -> tuple[float, Metrics]:
        """Compute reward for a precision level."""
        metrics: Metrics = {}

        # Compute geohash match
        pred_geohash = geohash_encode(pred_lat, pred_lon, precision=level.geohash_chars)
        gt_geohash_truncated = self.gt_geohash[:level.geohash_chars]

        prefix_match = common_prefix_len(pred_geohash, gt_geohash_truncated)
        geohash_reward = prefix_match / level.geohash_chars

        # Distance for metrics
        distance_km = haversine_km(pred_lat, pred_lon, self.ground_truth.lat, self.ground_truth.lon)

        # Improvement shaping: bounded by final score, so sandbagging can't increase total.
        improvement_bonus = 0.0
        if self.predictions:
            prev_lat, prev_lon = self.predictions[-1]
            prev_distance = haversine_km(prev_lat, prev_lon, self.ground_truth.lat, self.ground_truth.lon)
            score = geoguessr_score(distance_km) / 5000.0
            prev_score = geoguessr_score(prev_distance) / 5000.0
            improvement_bonus = float(self.config.improvement_bonus) * max(0.0, score - prev_score)

        # Total reward for this level
        reward = geohash_reward + improvement_bonus

        # Metrics
        metrics[f"geohash/{level.name}_match"] = float(prefix_match)
        metrics[f"geohash/{level.name}_reward"] = geohash_reward
        metrics[f"distance/{level.name}_km"] = distance_km
        metrics["improvement_bonus"] = improvement_bonus

        return reward, metrics

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Build initial observation with image and first prompt."""
        level = self.config.levels[0]
        prompt = self._get_prompt_for_level(level)

        content = [
            ImagePart(type="image", image=self.image),
            TextPart(type="text", text=prompt),
        ]
        self.messages = [Message(role="user", content=content)]

        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Process model response and advance to next level or finish."""
        message, parse_success = self.renderer.parse_response(action)
        response_text = ensure_text(message["content"])
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        level = self.current_level
        if level is None:
            # Shouldn't happen, but handle gracefully
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"error": 1.0},
            )

        # Parse coordinates from response
        parsed = parse_geo_response(response_text)

        if parsed.location is None:
            # Failed to parse - give format penalty and end episode
            return StepResult(
                reward=-0.1,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"format_error": 1.0, "level": self.current_level_idx},
            )

        pred_lat, pred_lon = parsed.location.lat, parsed.location.lon

        # Compute reward for this level (BEFORE appending to predictions)
        level_reward, level_metrics = self._compute_level_reward(level, pred_lat, pred_lon)

        # Now append to predictions for next turn's comparison
        self.predictions.append((pred_lat, pred_lon))

        # Weight the reward
        weighted_reward = level.reward_weight * level_reward
        self.level_rewards[level.name] = level_reward

        # Add level info to metrics
        level_metrics["level"] = self.current_level_idx
        level_metrics["level_name"] = level.name
        level_metrics[f"reward/{level.name}"] = level_reward
        level_metrics[f"reward/{level.name}_weighted"] = weighted_reward

        # Advance level
        self.current_level_idx += 1
        episode_done = self.current_level_idx >= len(self.config.levels)

        if episode_done:
            # Compute total reward
            total_reward = sum(
                lvl.reward_weight * self.level_rewards.get(lvl.name, 0)
                for lvl in self.config.levels
            )
            level_metrics["reward/total"] = total_reward

            # Final distance
            final_distance = haversine_km(pred_lat, pred_lon, self.ground_truth.lat, self.ground_truth.lon)
            level_metrics["distance/final_km"] = final_distance
            level_metrics["distance_bucket"] = distance_bucket(final_distance)

            next_obs = tinker.ModelInput.empty()
        else:
            # Build next level observation
            self.messages.append(Message(role="assistant", content=response_text))

            next_level = self.config.levels[self.current_level_idx]
            next_prompt = self._get_prompt_for_level(next_level)
            self.messages.append(Message(role="user", content=next_prompt))

            next_obs = self.renderer.build_generation_prompt(self.messages)

        return StepResult(
            reward=weighted_reward,
            episode_done=episode_done,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics=level_metrics,
        )


# -----------------------------------------------------------------------------
# Group Builder (for GRPO)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GeohashCurriculumGroupBuilder(EnvGroupBuilder):
    """Builder for groups of GeohashCurriculumEnv instances."""

    env_thunk: Callable[[], GeohashCurriculumEnv]
    num_envs: int
    dataset_name: str = "geospot-curriculum"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]

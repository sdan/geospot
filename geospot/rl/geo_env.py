"""
Geolocation RL environment for VLM training.
"""

import logging
from dataclasses import dataclass
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
)
from geospot.renderers import (
    ImagePart,
    Message,
    Renderer,
    TextPart,
    ensure_text,
)
from geospot.completers import StopCondition

logger = logging.getLogger(__name__)


DEFAULT_GEO_PROMPT = """Where is this? Reply with coordinates.

Latitude: <degrees>
Longitude: <degrees>"""


@dataclass
class GeoEnvConfig:
    prompt_template: str = DEFAULT_GEO_PROMPT
    reward_config: GeoRewardConfig | None = None
    max_image_size: int = 480
    format_coef: float = 0.1


class GeoEnv(Env):
    """Single-turn environment: image -> location prediction -> reward."""

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoLocation,
        renderer: Renderer,
        config: GeoEnvConfig | None = None,
    ):
        self.image = image
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.config = config or GeoEnvConfig()

        if self.config.max_image_size:
            self.image = self._resize_image(self.image, self.config.max_image_size)

        self._step_called = False

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

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        content = [
            ImagePart(type="image", image=self.image),
            TextPart(type="text", text=self.config.prompt_template),
        ]
        messages = [Message(role="user", content=content)]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        if self._step_called:
            raise RuntimeError("GeoEnv.step() can only be called once")
        self._step_called = True

        message, parse_success = self.renderer.parse_response(action)
        response_text = ensure_text(message["content"])
        parsed = parse_geo_response(response_text)

        reward_result = compute_geo_reward(
            prediction=parsed,
            ground_truth=self.ground_truth,
            config=self.config.reward_config,
        )

        # Format penalty (like ProblemEnv)
        format_valid = float(parse_success and reward_result.format_valid)
        total_reward = self.config.format_coef * (format_valid - 1) + reward_result.total_reward

        metrics = reward_result.to_metrics()
        metrics["format"] = format_valid
        if reward_result.distance_km is not None:
            metrics["distance_bucket"] = distance_bucket(reward_result.distance_km)

        logger.debug(
            f"GeoEnv: distance={reward_result.distance_km:.1f}km, reward={total_reward:.3f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


@dataclass(frozen=True)
class GeoGroupBuilder(EnvGroupBuilder):
    """Builder for groups of GeoEnv instances (for GRPO)."""

    env_thunk: Callable[[], GeoEnv]
    num_envs: int
    dataset_name: str = "geospot"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]

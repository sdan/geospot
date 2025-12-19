"""
Telescoping multi-turn geolocation RL environment.

All turns predict coordinates. Rewards are *telescoping* (potential-based) so the
episode return equals the final distance score:

  r_t = S(d_t) - S(d_{t-1})

where S(d) is a bounded distance score (default: GeoGuessr-style score in [0, 1]).

This provides dense credit assignment without changing the optimal policy for the
final objective (maximize S(d_final)).
"""

import logging
import math
import re
from dataclasses import dataclass, field
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
    distance_bucket,
    geoguessr_score,
    haversine_km,
    parse_geo_response,
)
from geospot.renderers import ImagePart, Message, Renderer, TextPart, ensure_text
from geospot.completers import StopCondition

logger = logging.getLogger(__name__)


COARSE_PROMPT = """Where is this location? Give an approximate estimate.

Think about what continent and general region this might be.

Latitude: <degrees>
Longitude: <degrees>"""

REFINE_PROMPT_1DP = """Your last guess was approximately ({prev_lat:.1f}, {prev_lon:.1f}).

Now refine your estimate. Consider country and region-level features.

Latitude: <degrees>
Longitude: <degrees>"""

REFINE_PROMPT_2DP = """Your last guess was ({prev_lat:.2f}, {prev_lon:.2f}).

Now give your final, most precise coordinates. Look for city-level and local features.

Latitude: <degrees>
Longitude: <degrees>"""


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@dataclass(frozen=True)
class TelescopingTurn:
    name: str
    prompt_template: str


DEFAULT_TURNS = [
    TelescopingTurn("coarse", COARSE_PROMPT),
    TelescopingTurn("refine", REFINE_PROMPT_1DP),
    TelescopingTurn("final", REFINE_PROMPT_2DP),
]


@dataclass
class TelescopingGeoEnvConfig:
    turns: list[TelescopingTurn] = field(default_factory=lambda: DEFAULT_TURNS.copy())
    max_image_size: int = 512

    # Reward: S(d) is either "geoguessr" (default) or "exp".
    score_kind: str = "geoguessr"
    exp_tau_km: float = 2000.0

    # Penalty when output can't be parsed into coordinates.
    format_penalty: float = 0.1


class TelescopingGeoEnv(Env):
    """
    Multi-turn coords-only environment with telescoping distance rewards.

    Each turn asks for coordinates; later turns show the model's previous guess.
    """

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoLocation,
        renderer: Renderer,
        config: TelescopingGeoEnvConfig | None = None,
    ):
        self.image = image
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.config = config or TelescopingGeoEnvConfig()

        if self.config.max_image_size:
            self.image = self._resize_image(self.image, self.config.max_image_size)

        self.current_turn_idx = 0
        self.prev_score = 0.0
        self.prev_pred: tuple[float, float] | None = None
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

    def _score(self, distance_km: float) -> float:
        if self.config.score_kind == "geoguessr":
            return geoguessr_score(distance_km) / 5000.0
        if self.config.score_kind == "exp":
            tau = max(1e-6, float(self.config.exp_tau_km))
            return float(math.exp(-distance_km / tau))
        raise ValueError(f"Unknown score_kind: {self.config.score_kind}")

    def _prompt_for_turn(self, turn: TelescopingTurn) -> str:
        if self.prev_pred is None:
            return turn.prompt_template
        try:
            return turn.prompt_template.format(prev_lat=self.prev_pred[0], prev_lon=self.prev_pred[1])
        except KeyError:
            return turn.prompt_template

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        turn = self.config.turns[0]
        prompt = self._prompt_for_turn(turn)
        content = [
            ImagePart(type="image", image=self.image),
            TextPart(type="text", text=prompt),
        ]
        self.messages = [Message(role="user", content=content)]
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        response_text = ensure_text(message["content"])

        if self.current_turn_idx >= len(self.config.turns):
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"error": 1.0},
            )

        parsed = parse_geo_response(response_text)
        if parsed.location is None:
            return StepResult(
                reward=-float(self.config.format_penalty),
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": 0.0,
                    "format_error": 1.0,
                    "turn": self.current_turn_idx,
                },
            )

        pred_lat, pred_lon = parsed.location.lat, parsed.location.lon
        distance_km = haversine_km(pred_lat, pred_lon, self.ground_truth.lat, self.ground_truth.lon)
        score = self._score(distance_km)

        reward = float(score - self.prev_score)
        self.prev_score = float(score)
        self.prev_pred = (pred_lat, pred_lon)

        turn_name = self.config.turns[self.current_turn_idx].name
        metrics: Metrics = {
            "turn": self.current_turn_idx,
            "turn_name": turn_name,
            "format": float(bool(parse_success and parsed.format_valid)),
            f"distance/{turn_name}_km": float(distance_km),
            f"score/{turn_name}": float(score),
            f"reward/{turn_name}": float(reward),
        }

        self.current_turn_idx += 1
        episode_done = self.current_turn_idx >= len(self.config.turns)

        if episode_done:
            metrics["distance/final_km"] = float(distance_km)
            metrics["distance_bucket"] = distance_bucket(distance_km)
            metrics["score/final"] = float(score)
            metrics["reward/total"] = float(score)
            next_obs = tinker.ModelInput.empty()
        else:
            self.messages.append(Message(role="assistant", content=_strip_think(parsed.raw_text)))
            next_turn = self.config.turns[self.current_turn_idx]
            self.messages.append(Message(role="user", content=self._prompt_for_turn(next_turn)))
            next_obs = self.renderer.build_generation_prompt(self.messages)

        return StepResult(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


@dataclass(frozen=True)
class TelescopingGeoGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], TelescopingGeoEnv]
    num_envs: int
    dataset_name: str = "geospot-telescoping"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


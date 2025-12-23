"""
Core RL types for GRPO training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, TypeAlias

import tinker

# Type aliases
Action: TypeAlias = list[int]
Observation: TypeAlias = tinker.ModelInput
StopCondition: TypeAlias = list[str] | list[int]
Metrics: TypeAlias = dict[str, float | int | str]


@dataclass
class TokensWithLogprobs:
    """Action tokens with their log probabilities."""
    tokens: list[int]
    logprobs: list[float]


@dataclass
class StepResult:
    """Result of taking an action in an environment."""
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)


@dataclass
class Transition:
    """A single (observation, action, reward) tuple."""
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """Sequence of transitions from a single episode."""
    transitions: list[Transition]
    final_ob: Observation


@dataclass
class TrajectoryGroup:
    """Group of trajectories for GRPO advantage computation."""
    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]
    metrics_G: list[Metrics]

    def get_total_rewards(self) -> list[float]:
        """Sum of per-step rewards + final reward for each trajectory."""
        return [
            sum(t.reward for t in traj.transitions) + final
            for traj, final in zip(self.trajectories_G, self.final_rewards_G, strict=True)
        ]


class Env(ABC):
    """Stateful environment for a single episode."""

    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        pass

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        pass


class EnvGroupBuilder(ABC):
    """Builds a group of environments for GRPO (rewards centered across group)."""

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        pass

    async def compute_group_rewards(
        self, trajectories: list[Trajectory], envs: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute final reward for each trajectory (default: 0)."""
        return [(0.0, {}) for _ in trajectories]

    def logging_tags(self) -> list[str]:
        return []

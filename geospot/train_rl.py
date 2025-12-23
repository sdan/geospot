"""
GRPO (Group Relative Policy Optimization) training for visual geolocation.

Algorithm:
1. Sample N trajectories per image (group_size)
2. Compute rewards using distance-based scoring
3. Center advantages within each group: advantage_i = reward_i - mean(rewards)
4. Update policy using importance-weighted gradient

Run:
    uv run python -m geospot.train_rl hf_repo=osv5m/osv5m
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import Literal, Sequence

import chz
import tinker
import wandb
from tinker.types import AdamParams

from geospot.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from geospot.datasets import iterate_samples, GeoSample
from geospot.envs import (
    # Single-turn
    SingleTurnGeoEnv,
    SingleTurnGeoEnvConfig,
    SingleTurnGeoGroupBuilder,
    # Multi-turn
    MultiTurnGeoEnv,
    MultiTurnGeoEnvConfig,
    MultiTurnGeoGroupBuilder,
    # Utilities
    GeoLocation,
)
from geospot.renderers import get_renderer
from geospot.types import Env, EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition

logger = logging.getLogger(__name__)


# =============================================================================
# Inlined utilities (from tokenizer_utils, image_processing_utils, completers)
# =============================================================================


@cache
def get_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    kwargs = {"trust_remote_code": True} if "qwen" in model_name.lower() else {}
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


@cache
def get_image_processor(model_name: str):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_name, use_fast=True)


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    logprobs: list[float]


@dataclass
class TinkerTokenCompleter:
    """Sample actions from Tinker model with logprobs."""
    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: list[str] | list[int]
    ) -> TokensWithLogprobs:
        result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop, max_tokens=self.max_tokens, temperature=self.temperature
            ),
        )
        seq = result.sequences[0]
        assert seq.logprobs is not None
        return TokensWithLogprobs(tokens=seq.tokens, logprobs=seq.logprobs)


# =============================================================================
# Inlined rollouts (from rollouts.py)
# =============================================================================


async def do_single_rollout(policy: TinkerTokenCompleter, env: Env) -> Trajectory:
    """Run environment until episode done, collecting transitions."""
    transitions = []
    ob, stop = await env.initial_observation()
    while True:
        ac = await policy(ob, stop)
        step = await env.step(ac.tokens)
        transitions.append(Transition(
            ob=ob, ac=ac, reward=step.reward,
            episode_done=step.episode_done, metrics=step.metrics
        ))
        ob, stop = step.next_observation, step.next_stop_condition
        if step.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


async def do_group_rollout(
    builder: EnvGroupBuilder, policy: TinkerTokenCompleter
) -> TrajectoryGroup:
    """Run rollouts for all envs in group, compute rewards."""
    envs: Sequence[Env] = await builder.make_envs()
    trajectories = await asyncio.gather(*[do_single_rollout(policy, e) for e in envs])
    rewards_and_metrics = await builder.compute_group_rewards(trajectories, envs)
    rewards, metrics = zip(*rewards_and_metrics, strict=True)
    return TrajectoryGroup(trajectories, list(rewards), list(metrics))


# =============================================================================
# Training
# =============================================================================


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    """Remove mask field before training (used for logging only)."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


LogdirBehavior = Literal["delete", "resume", "ask", "raise"]


@chz.chz
class Config:
    """GRPO training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str = "qwen3_vl"

    # Data
    hf_repo: str = "osv5m/osv5m"
    max_steps: int = 100
    shuffle_buffer: int = 1000

    # Training
    batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 4e-5
    max_tokens: int = 128
    temperature: float = 1.0

    # Environment: "single" or "multi"
    env_type: str = "single"
    format_penalty: float = 0.1
    max_image_size: int = 512
    # Multi-turn reward weights (only used if env_type="multi")
    country_reward: float = 0.2
    region_reward: float = 0.2
    city_reward: float = 0.2

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-rl"

    # Checkpointing
    save_every: int = 25
    load_checkpoint_path: str | None = None

    # Misc
    seed: int = 0
    base_url: str | None = None
    behavior_if_log_dir_exists: LogdirBehavior = "ask"


def _sample_to_location(sample: GeoSample) -> GeoLocation:
    """Convert GeoSample to GeoLocation with all OSV-5M labels."""
    return GeoLocation(
        lat=sample.lat,
        lon=sample.lon,
        country=sample.country,
        region=sample.region,
        sub_region=sample.sub_region,
        city=sample.city,
    )


def _build_single_turn_group(
    sample: GeoSample, renderer, config: SingleTurnGeoEnvConfig, group_size: int
) -> SingleTurnGeoGroupBuilder:
    """Build env group for single-turn training."""
    def make_env():
        return SingleTurnGeoEnv(
            image=sample.image,
            ground_truth=_sample_to_location(sample),
            renderer=renderer,
            config=config,
        )
    return SingleTurnGeoGroupBuilder(
        env_thunk=make_env,
        num_envs=group_size,
        dataset_name=sample.source or "geospot",
    )


def _build_multi_turn_group(
    sample: GeoSample, renderer, config: MultiTurnGeoEnvConfig, group_size: int
) -> MultiTurnGeoGroupBuilder:
    """Build env group for multi-turn training."""
    def make_env():
        return MultiTurnGeoEnv(
            image=sample.image,
            ground_truth=_sample_to_location(sample),
            renderer=renderer,
            config=config,
        )
    return MultiTurnGeoGroupBuilder(
        env_thunk=make_env,
        num_envs=group_size,
        dataset_name=sample.source or "geospot",
    )


async def run_training(cfg: Config):
    """Main GRPO training loop."""
    if cfg.env_type not in ("single", "multi"):
        raise ValueError(f"env_type must be 'single' or 'multi', got: {cfg.env_type}")

    # Setup log path
    if cfg.log_path:
        log_path = cfg.log_path
    else:
        model = cfg.model_name.replace("/", "-")
        log_path = f"/tmp/geospot-rl/{model}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    # Handle existing log dir
    if os.path.exists(log_path):
        if cfg.behavior_if_log_dir_exists == "delete":
            shutil.rmtree(log_path)
        elif cfg.behavior_if_log_dir_exists == "raise":
            raise ValueError(f"Log dir exists: {log_path}")
        elif cfg.behavior_if_log_dir_exists == "ask":
            resp = input(f"Log dir {log_path} exists. [delete/resume/exit]: ")
            if resp == "delete":
                shutil.rmtree(log_path)
            elif resp == "exit":
                return
    os.makedirs(log_path, exist_ok=True)

    logger.info(f"GRPO Training: {cfg.hf_repo} -> {log_path}")
    logger.info(f"Model: {cfg.model_name}, batch={cfg.batch_size}, group={cfg.group_size}")

    if cfg.wandb_project:
        # Descriptive run name: rl-single-qwen30b-osv5m or rl-multi-qwen30b-osv5m
        model_short = cfg.model_name.split("/")[-1].lower().replace("-instruct", "")
        dataset_short = cfg.hf_repo.split("/")[-1]
        run_name = f"rl-{cfg.env_type}-{model_short}-{dataset_short}"
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            tags=["grpo", cfg.env_type, "rl"],
            config=vars(cfg) if hasattr(cfg, '__dict__') else {},
        )

    # Setup model components
    tokenizer = get_tokenizer(cfg.model_name)
    image_processor = get_image_processor(cfg.model_name)
    renderer = get_renderer(cfg.renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    # Create env config based on type
    if cfg.env_type == "single":
        env_config = SingleTurnGeoEnvConfig(
            max_image_size=cfg.max_image_size,
            format_penalty=cfg.format_penalty,
        )
    else:  # multi
        env_config = MultiTurnGeoEnvConfig(
            max_image_size=cfg.max_image_size,
            format_penalty=cfg.format_penalty,
            country_reward=cfg.country_reward,
            region_reward=cfg.region_reward,
            city_reward=cfg.city_reward,
        )

    sample_iter = iterate_samples(
        hf_repo=cfg.hf_repo, seed=cfg.seed, shuffle_buffer=cfg.shuffle_buffer
    )

    # Setup training client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if cfg.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded checkpoint: {cfg.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    adam = AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # Training loop
    for step in range(cfg.max_steps):
        t0 = time.time()

        # Get sampling client with current weights
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"{step:06d}"
        )
        policy = TinkerTokenCompleter(
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

        # Collect samples
        samples: list[GeoSample] = []
        for _ in range(cfg.batch_size):
            try:
                samples.append(next(sample_iter))
            except StopIteration:
                logger.info("Dataset exhausted, restarting...")
                sample_iter = iterate_samples(
                    hf_repo=cfg.hf_repo, seed=cfg.seed + step, shuffle_buffer=cfg.shuffle_buffer
                )
                samples.append(next(sample_iter))

        # Build env groups and run rollouts
        if cfg.env_type == "single":
            builders = [_build_single_turn_group(s, renderer, env_config, cfg.group_size) for s in samples]
        else:
            builders = [_build_multi_turn_group(s, renderer, env_config, cfg.group_size) for s in samples]
        groups = await asyncio.gather(*[do_group_rollout(b, policy) for b in builders])
        groups = remove_constant_reward_groups(groups)

        # Compute GRPO advantages and assemble training data
        advantages = compute_advantages(groups)
        datums, _ = assemble_training_data(groups, advantages)

        if not datums:
            logger.warning(f"Step {step}: no training datums")
            continue

        # Train
        datums_clean = [_remove_mask(d) for d in datums]
        fwd_bwd = training_client.forward_backward(datums_clean, loss_fn="importance_sampling")
        optim = training_client.optim_step(adam)
        fwd_bwd.result()
        optim.result()

        # Metrics
        rewards = [sum(g.get_total_rewards()) / len(g.get_total_rewards()) for g in groups]
        distances = []
        for g in groups:
            for t in g.trajectories_G:
                for tr in reversed(t.transitions):
                    if "distance_km" in tr.metrics:
                        distances.append(float(tr.metrics["distance_km"]))
                        break

        mean_reward = sum(rewards) / len(rewards) if rewards else 0
        mean_dist = sum(distances) / len(distances) if distances else 0
        elapsed = time.time() - t0
        skipped = max(len(builders) - len(groups), 0)

        logger.info(
            f"Step {step}: reward={mean_reward:.3f}, dist={mean_dist:.0f}km, "
            f"datums={len(datums_clean)}, skipped={skipped}, time={elapsed:.1f}s"
        )

        if cfg.wandb_project:
            wandb.log({
                "step": step, "reward/mean": mean_reward, "distance_km/mean": mean_dist,
                "datums": len(datums_clean), "skipped_uniform": skipped, "time_s": elapsed,
            })

        if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            training_client.save_state(name=f"step_{step:06d}").result()
            logger.info(f"Saved checkpoint: step_{step:06d}")

    # Final save
    result = training_client.save_state(name="final").result()
    logger.info(f"Training complete! Checkpoint: {result.path}")
    if cfg.wandb_project:
        wandb.finish()


def main(cfg: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_training(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

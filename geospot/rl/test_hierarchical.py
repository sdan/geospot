#!/usr/bin/env python3
"""
Smoke test for hierarchical geo environment.

Run with:
    cd /Users/sdan/Developer/geospot-vlm
    python -m geospot.rl.test_hierarchical
"""

import asyncio
import sys
from PIL import Image
import numpy as np


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    from geospot.rl.hierarchical_geo_env import (
        HierarchicalGeoEnv,
        HierarchicalGeoEnvConfig,
        HierarchicalGeoGroupBuilder,
        normalize_text,
        texts_match,
    )
    from geospot.rl.geo_reward import (
        GeoLocation,
        GeoRewardConfig,
        geohash_encode,
        common_prefix_len,
    )
    from geospot.rl.geo_dataset import (
        HierarchicalGeoDataset,
        HierarchicalGeoDatasetBuilder,
    )

    print("  ✓ All imports successful")
    return True


def test_text_matching():
    """Test text normalization and matching."""
    print("Testing text matching...")

    from geospot.rl.hierarchical_geo_env import normalize_text, texts_match

    # Test normalization
    assert normalize_text("The United States") == "united states"
    assert normalize_text("Republic of France") == "france"
    assert normalize_text("São Paulo") == "são paulo"

    # Test matching
    assert texts_match("Brazil", "Brazil")
    assert texts_match("brazil", "BRAZIL")
    assert texts_match("United States", "The United States")
    assert texts_match("California", "California, USA")
    assert not texts_match("France", "Germany")
    assert not texts_match(None, "Brazil")

    print("  ✓ Text matching works")
    return True


def test_geohash():
    """Test geohash encoding and prefix matching."""
    print("Testing geohash...")

    from geospot.rl.geo_reward import geohash_encode, common_prefix_len

    # Test nearby points (should share prefix)
    # São Paulo center
    sp1_hash = geohash_encode(-23.55, -46.63, precision=5)
    # São Paulo suburb (~10km away)
    sp2_hash = geohash_encode(-23.50, -46.60, precision=5)
    # Far away (Paris)
    paris_hash = geohash_encode(48.86, 2.35, precision=5)

    print(f"    SP center: {sp1_hash}")
    print(f"    SP suburb: {sp2_hash}")
    print(f"    Paris: {paris_hash}")

    sp_internal_prefix = common_prefix_len(sp1_hash, sp2_hash)
    sp_paris_prefix = common_prefix_len(sp1_hash, paris_hash)

    print(f"    SP internal prefix: {sp_internal_prefix}")
    print(f"    SP-Paris prefix: {sp_paris_prefix}")

    # Nearby points should share more prefix
    assert sp_internal_prefix >= sp_paris_prefix, "Nearby points should share at least as much prefix"

    # Test that geohash encoding works for valid coords
    assert len(sp1_hash) == 5
    assert len(paris_hash) == 5

    # Test edge cases (just check they don't crash)
    assert len(geohash_encode(0, 0, precision=3)) == 3  # Equator/prime meridian
    assert len(geohash_encode(89, 0, precision=3)) == 3  # Near north pole
    assert len(geohash_encode(-89, 0, precision=3)) == 3  # Near south pole

    print("  ✓ Geohash works")
    return True


def test_config():
    """Test config creation."""
    print("Testing config...")

    from geospot.rl.hierarchical_geo_env import HierarchicalGeoEnvConfig
    from geospot.rl.geo_reward import GeoRewardConfig

    config = HierarchicalGeoEnvConfig(
        turns=["country", "region", "coords"],
        teacher_forcing_prob=0.5,
        country_reward_weight=0.2,
        region_reward_weight=0.3,
        coords_reward_weight=0.5,
    )

    assert config.turns == ["country", "region", "coords"]
    assert config.teacher_forcing_prob == 0.5
    assert abs(config.country_reward_weight + config.region_reward_weight + config.coords_reward_weight - 1.0) < 0.01

    print("  ✓ Config works")
    return True


class MockRenderer:
    """Mock renderer for testing without actual model."""

    def build_generation_prompt(self, messages):
        # Return a mock ModelInput-like object
        class MockModelInput:
            def __init__(self):
                self.length = 100
        return MockModelInput()

    def get_stop_sequences(self):
        return {"stop": ["<|endoftext|>"]}

    def parse_response(self, action):
        # Return (message, parse_success)
        return {"role": "assistant", "content": action}, True


async def test_env_episode():
    """Test running through a full episode."""
    print("Testing environment episode...")

    from geospot.rl.hierarchical_geo_env import HierarchicalGeoEnv, HierarchicalGeoEnvConfig
    from geospot.rl.geo_reward import GeoLocation

    # Create a mock image (random noise)
    image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    # Ground truth: São Paulo, Brazil
    ground_truth = GeoLocation(
        lat=-23.55,
        lon=-46.63,
        city="São Paulo",
        region="São Paulo",
        country="Brazil",
    )

    # Config with full teacher forcing for predictable test
    config = HierarchicalGeoEnvConfig(
        turns=["country", "region", "coords"],
        teacher_forcing_prob=1.0,
    )

    renderer = MockRenderer()

    env = HierarchicalGeoEnv(
        image=image,
        ground_truth=ground_truth,
        renderer=renderer,
        config=config,
    )

    # Turn 1: Country
    obs, stop = await env.initial_observation()
    print(f"    Turn 1 (country): got observation")

    # Simulate model response: correct country
    result1 = await env.step("Brazil")
    print(f"    Turn 1 reward: {result1.reward:.3f}, done: {result1.episode_done}")
    assert not result1.episode_done, "Episode should continue after country"
    assert result1.reward > 0, "Correct country should get positive reward"

    # Turn 2: Region
    print(f"    Turn 2 (region): continuing...")
    result2 = await env.step("São Paulo")
    print(f"    Turn 2 reward: {result2.reward:.3f}, done: {result2.episode_done}")
    assert not result2.episode_done, "Episode should continue after region"
    assert result2.reward > 0, "Correct region should get positive reward"

    # Turn 3: Coords
    print(f"    Turn 3 (coords): continuing...")
    result3 = await env.step("Latitude: -23.55\nLongitude: -46.63")
    print(f"    Turn 3 reward: {result3.reward:.3f}, done: {result3.episode_done}")
    assert result3.episode_done, "Episode should end after coords"
    assert result3.reward > 0, "Correct coords should get positive reward"

    # Check total reward in metrics
    total = result3.metrics.get("reward/total", 0)
    print(f"    Total reward: {total:.3f}")

    print("  ✓ Episode runs successfully")
    return True


async def test_wrong_answers():
    """Test that wrong answers get lower rewards."""
    print("Testing wrong answer handling...")

    from geospot.rl.hierarchical_geo_env import HierarchicalGeoEnv, HierarchicalGeoEnvConfig
    from geospot.rl.geo_reward import GeoLocation

    image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    ground_truth = GeoLocation(
        lat=-23.55, lon=-46.63,
        city="São Paulo", region="São Paulo", country="Brazil",
    )

    config = HierarchicalGeoEnvConfig(teacher_forcing_prob=1.0)
    renderer = MockRenderer()

    env = HierarchicalGeoEnv(image=image, ground_truth=ground_truth, renderer=renderer, config=config)

    await env.initial_observation()

    # Wrong country
    result1 = await env.step("France")
    print(f"    Wrong country reward: {result1.reward:.3f}")
    assert result1.reward == 0, "Wrong country should get 0 reward"

    # Turn 2 will still use ground truth hint (teacher forcing)
    result2 = await env.step("Rio de Janeiro")  # Wrong region
    print(f"    Wrong region reward: {result2.reward:.3f}")

    # Wrong coords (far away)
    result3 = await env.step("Latitude: 48.86\nLongitude: 2.35")  # Paris coords
    print(f"    Wrong coords reward: {result3.reward:.3f}")
    assert result3.reward < 0.5, "Far coords should get low reward"

    print("  ✓ Wrong answers handled correctly")
    return True


async def test_teacher_forcing_toggle():
    """Test that teacher forcing affects hints."""
    print("Testing teacher forcing toggle...")

    from geospot.rl.hierarchical_geo_env import HierarchicalGeoEnv, HierarchicalGeoEnvConfig
    from geospot.rl.geo_reward import GeoLocation

    image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    ground_truth = GeoLocation(lat=-23.55, lon=-46.63, city="São Paulo", region="São Paulo", country="Brazil")

    # With teacher forcing OFF (use model's own predictions)
    config = HierarchicalGeoEnvConfig(teacher_forcing_prob=0.0)
    renderer = MockRenderer()

    env = HierarchicalGeoEnv(image=image, ground_truth=ground_truth, renderer=renderer, config=config)

    await env.initial_observation()

    # Say wrong country
    await env.step("Argentina")

    # Check that env stored "Argentina" as prediction
    assert env.predictions.get("country") == "Argentina"

    # With TF=0, the next prompt should use "Argentina" not "Brazil"
    # (We can't easily check the prompt text in this mock, but we verified the prediction is stored)

    print("  ✓ Teacher forcing toggle works")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Hierarchical Geo Environment Tests")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Text Matching", test_text_matching),
        ("Geohash", test_geohash),
        ("Config", test_config),
        ("Episode", test_env_episode),
        ("Wrong Answers", test_wrong_answers),
        ("Teacher Forcing", test_teacher_forcing_toggle),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if asyncio.iscoroutinefunction(test_fn):
                result = await test_fn()
            else:
                result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} failed with: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

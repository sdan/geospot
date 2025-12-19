#!/usr/bin/env python3
"""
Smoke test for the telescoping geo environment.

Run with:
    cd /Users/sdan/Developer/geospot-vlm
    python -m geospot.rl.test_telescoping
"""

import asyncio
import numpy as np
from PIL import Image


class MockRenderer:
    """Mock renderer for testing without an actual tokenizer/model."""

    def build_generation_prompt(self, messages):
        class MockModelInput:
            def __init__(self):
                self.length = 100

        return MockModelInput()

    def get_stop_sequences(self):
        return {"stop": ["<|endoftext|>"]}

    def parse_response(self, action):
        return {"role": "assistant", "content": action}, True


async def test_env_episode():
    from geospot.rl.geo_reward import GeoLocation
    from geospot.rl.telescoping_geo_env import TelescopingGeoEnv, TelescopingGeoEnvConfig

    image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    ground_truth = GeoLocation(lat=-23.55, lon=-46.63)

    env = TelescopingGeoEnv(
        image=image,
        ground_truth=ground_truth,
        renderer=MockRenderer(),
        config=TelescopingGeoEnvConfig(score_kind="geoguessr"),
    )

    _obs, _stop = await env.initial_observation()

    r1 = await env.step("Latitude: -25\nLongitude: -50")
    assert not r1.episode_done
    assert 0.0 <= r1.reward <= 1.0

    r2 = await env.step("Latitude: -23.8\nLongitude: -46.8")
    assert not r2.episode_done

    r3 = await env.step("Latitude: -23.55\nLongitude: -46.63")
    assert r3.episode_done

    total = r1.reward + r2.reward + r3.reward
    assert abs(total - float(r3.metrics.get("reward/total", -999))) < 1e-6
    assert r3.metrics.get("distance/final_km", 1e9) < r1.metrics.get("distance/coarse_km", 0.0)


def main() -> int:
    asyncio.run(test_env_episode())
    print("✓ TelescopingGeoEnv smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


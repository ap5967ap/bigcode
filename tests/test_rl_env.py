from gymnasium.utils.env_checker import check_env

from rl.env import NightRouteEnv


def test_night_route_env_passes_check_and_steps() -> None:
    env = NightRouteEnv()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=42)
    assert obs.shape == (7,)
    assert info["valid_actions"] >= 0
    obs, reward, terminated, truncated, step_info = env.step(0)
    assert obs.shape == (7,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "valid_actions" in step_info

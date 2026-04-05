from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import osmnx as ox
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import tqdm

from classifier.archetype_classifier import (
    ADAPTIVE,
    ARCHETYPE_NAMES,
    COMFORT_SEEKER,
    EFFICIENCY_FIRST,
    VULNERABLE_SOLO,
)
from rl.agents import QLearningRouteAgent, agent_path_for_archetype
from rl.env import NightRouteEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "scored_graph.graphml"
TRAINING_CURVES_PATH = PROJECT_ROOT / "data" / "processed" / "training_curves.json"
RANDOM_SEED = 42
TOTAL_TIMESTEPS = 100_000
N_ENVS = 4
MIN_PPO_FPS = 350.0
Q_SUBGRAPH_NODES = 500
Q_EPISODES = 2000
Q_ALPHA = 0.2
Q_GAMMA = 0.95
Q_EPSILON_START = 0.35
Q_EPSILON_END = 0.05


class RouteMetricsCallback(BaseCallback):
    def __init__(self, archetype_name: str, total_timesteps: int) -> None:
        super().__init__()
        self.archetype_name = archetype_name
        self.total_timesteps = total_timesteps
        self.episode_rewards: list[float] = []
        self.episode_safety: list[float] = []
        self.episode_travel_time: list[float] = []
        self.snapshots: list[dict[str, float]] = []
        self._progress: tqdm | None = None

    def _on_training_start(self) -> None:
        self._progress = tqdm(total=self.total_timesteps, desc=f"PPO {self.archetype_name}")

    def _on_step(self) -> bool:
        if self._progress is not None:
            self._progress.update(self.training_env.num_envs)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for done, info, reward in zip(dones, infos, rewards, strict=False):
            if not done:
                continue
            self.episode_rewards.append(float(info.get("episode", {}).get("r", reward)))
            self.episode_safety.append(float(info.get("mean_path_safety", 0.0)))
            self.episode_travel_time.append(float(info.get("travel_time_sum", 0.0)))
        if self.num_timesteps % 5000 < self.training_env.num_envs and self.episode_rewards:
            self.snapshots.append(
                {
                    "timesteps": float(self.num_timesteps),
                    "mean_episode_reward": float(np.mean(self.episode_rewards[-100:])),
                    "mean_safety_score": float(np.mean(self.episode_safety[-100:])),
                    "mean_travel_time": float(np.mean(self.episode_travel_time[-100:])),
                }
            )
        return True

    def _on_training_end(self) -> None:
        if self._progress is not None:
            self._progress.close()


def _make_env(archetype_id: int, seed_offset: int) -> Callable[[], Monitor]:
    def _factory() -> Monitor:
        env = NightRouteEnv(graph_path=SCORED_GRAPH_PATH, archetype_id=archetype_id, seed=RANDOM_SEED + seed_offset)
        return Monitor(env)

    return _factory


def _largest_subgraph(graph: nx.MultiDiGraph, max_nodes: int) -> nx.MultiDiGraph:
    undirected = graph.to_undirected()
    largest_component = max(nx.connected_components(undirected), key=len)
    selected_nodes = list(largest_component)[:max_nodes]
    return graph.subgraph(selected_nodes).copy()


def _q_state_key(node_id: Any, destination_node_id: Any, archetype_id: int) -> str:
    return f"{node_id}|{destination_node_id}|{archetype_id}"


def _train_q_agent(archetype_id: int) -> dict[str, Any]:
    logger.warning(f"Switching {ARCHETYPE_NAMES[archetype_id]} to Q-learning fallback")
    base_graph = ox.load_graphml(SCORED_GRAPH_PATH)
    subgraph = _largest_subgraph(base_graph, Q_SUBGRAPH_NODES)
    env = NightRouteEnv(graph_path=SCORED_GRAPH_PATH, archetype_id=archetype_id)
    env.graph = subgraph
    env.node_ids = list(subgraph.nodes())
    env.node_to_idx = {node_id: idx for idx, node_id in enumerate(env.node_ids)}
    env.idx_to_node = {idx: node_id for node_id, idx in env.node_to_idx.items()}
    env.out_edges = env._build_outgoing_edge_index()
    env.max_degree = max((len(edges) for edges in env.out_edges.values()), default=1)
    env.action_space = env.action_space.__class__(env.max_degree)
    env.node_xy = {node_id: (float(attrs["x"]), float(attrs["y"])) for node_id, attrs in subgraph.nodes(data=True)}
    env.candidate_nodes = [node for node, edges in env.out_edges.items() if edges]

    q_table: dict[str, np.ndarray] = {}
    episode_rewards: list[float] = []
    episode_safety: list[float] = []
    episode_time: list[float] = []
    rng = np.random.default_rng(RANDOM_SEED + archetype_id)

    for episode_idx in tqdm(range(Q_EPISODES), desc=f"Q-learning {ARCHETYPE_NAMES[archetype_id]}"):
        epsilon = Q_EPSILON_START + (Q_EPSILON_END - Q_EPSILON_START) * (episode_idx / max(Q_EPISODES - 1, 1))
        obs, info = env.reset(seed=RANDOM_SEED + episode_idx)
        done = False
        cumulative_reward = 0.0
        while not done:
            state_key = _q_state_key(env.current_node, env.destination_node, archetype_id)
            q_values = q_table.setdefault(state_key, np.zeros(env.max_degree, dtype=np.float32))
            valid_actions = max(1, info.get("valid_actions", len(env.out_edges.get(env.current_node, []))))
            if rng.random() < epsilon:
                action = int(rng.integers(0, valid_actions))
            else:
                action = int(np.argmax(q_values[:valid_actions]))
            _, reward, terminated, truncated, next_info = env.step(action)
            next_state_key = _q_state_key(env.current_node, env.destination_node, archetype_id)
            next_q = q_table.setdefault(next_state_key, np.zeros(env.max_degree, dtype=np.float32))
            td_target = reward if terminated else reward + Q_GAMMA * float(np.max(next_q[: max(1, next_info.get("valid_actions", 1))]))
            q_values[action] += Q_ALPHA * (td_target - q_values[action])
            cumulative_reward += reward
            info = next_info
            done = terminated or truncated
        episode_rewards.append(cumulative_reward)
        episode_safety.append(float(np.mean([float(edge.get("safety_score", 50.0)) for edge in env.path_edges])) if env.path_edges else 0.0)
        episode_time.append(float(sum(float(edge.get("travel_time", 0.0)) for edge in env.path_edges)))

    agent_path = agent_path_for_archetype(archetype_id)
    serialized_q = {state: values.tolist() for state, values in q_table.items()}
    QLearningRouteAgent.save(agent_path, serialized_q, archetype_id, env.max_degree)
    snapshots = [
        {
            "timesteps": float((idx + 1) * env.max_steps),
            "mean_episode_reward": float(np.mean(episode_rewards[max(0, idx - 99): idx + 1])),
            "mean_safety_score": float(np.mean(episode_safety[max(0, idx - 99): idx + 1])),
            "mean_travel_time": float(np.mean(episode_time[max(0, idx - 99): idx + 1])),
        }
        for idx in range(99, len(episode_rewards), 100)
    ]
    return {
        "algorithm": "q_learning",
        "agent_path": str(agent_path),
        "mean_episode_reward": float(np.mean(episode_rewards[-100:])),
        "mean_safety_score": float(np.mean(episode_safety[-100:])),
        "mean_travel_time": float(np.mean(episode_time[-100:])),
        "snapshots": snapshots,
    }


def _train_ppo_agent(archetype_id: int) -> dict[str, Any]:
    archetype_name = ARCHETYPE_NAMES[archetype_id]
    try:
        vec_env = SubprocVecEnv([_make_env(archetype_id, i) for i in range(N_ENVS)])
    except Exception as exc:
        logger.warning(f"SubprocVecEnv failed for {archetype_name}, falling back to DummyVecEnv: {exc}")
        vec_env = DummyVecEnv([_make_env(archetype_id, i) for i in range(N_ENVS)])

    callback = RouteMetricsCallback(archetype_name, TOTAL_TIMESTEPS)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=RANDOM_SEED,
        verbose=0,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    elapsed = max(time.perf_counter() - started, 1e-9)
    fps = TOTAL_TIMESTEPS / elapsed
    vec_env.close()

    if fps < MIN_PPO_FPS:
        logger.warning(
            f"PPO throughput {fps:.1f} fps below threshold {MIN_PPO_FPS:.1f}; retraining {archetype_name} with Q-learning fallback"
        )
        return _train_q_agent(archetype_id)

    agent_path = agent_path_for_archetype(archetype_id)
    model.save(agent_path)
    return {
        "algorithm": "ppo",
        "agent_path": str(agent_path),
        "fps": float(fps),
        "mean_episode_reward": float(np.mean(callback.episode_rewards[-100:])) if callback.episode_rewards else 0.0,
        "mean_safety_score": float(np.mean(callback.episode_safety[-100:])) if callback.episode_safety else 0.0,
        "mean_travel_time": float(np.mean(callback.episode_travel_time[-100:])) if callback.episode_travel_time else 0.0,
        "snapshots": callback.snapshots,
    }


def _ensure_zip_suffix(agent_path: Path) -> None:
    if agent_path.exists() and zipfile.is_zipfile(agent_path):
        return
    raise FileNotFoundError(f"Expected trained agent artifact at {agent_path}")


def train_all_agents() -> dict[str, dict[str, Any]]:
    if not SCORED_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing scored graph: {SCORED_GRAPH_PATH}")
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for archetype_id in [VULNERABLE_SOLO, COMFORT_SEEKER, EFFICIENCY_FIRST, ADAPTIVE]:
        archetype_name = ARCHETYPE_NAMES[archetype_id]
        logger.info(f"Training routing agent for {archetype_name}")
        try:
            result = _train_ppo_agent(archetype_id)
        except Exception as exc:
            logger.warning(f"PPO training failed for {archetype_name}: {exc}")
            result = _train_q_agent(archetype_id)
        agent_path = Path(result["agent_path"])
        _ensure_zip_suffix(agent_path)
        results[archetype_name] = result
        logger.info(
            f"{archetype_name} agent saved to {agent_path} "
            f"algo={result['algorithm']} mean_reward={result['mean_episode_reward']:.3f} "
            f"mean_safety={result['mean_safety_score']:.2f} mean_time={result['mean_travel_time']:.2f}"
        )

    TRAINING_CURVES_PATH.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved training curves to {TRAINING_CURVES_PATH}")
    return results


def main() -> None:
    train_all_agents()


if __name__ == "__main__":
    main()

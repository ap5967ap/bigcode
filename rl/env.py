from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
import osmnx as ox
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from loguru import logger

from classifier.archetype_classifier import ADAPTIVE, get_archetype_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "scored_graph.graphml"
FEATURED_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "featured_graph.graphml"
RANDOM_SEED = 42
MIN_OD_DISTANCE_METERS = 500.0


class NightRouteEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph_path: str | Path | None = None,
        archetype_id: int = ADAPTIVE,
        time_of_day: float = 21.0,
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__()
        selected_graph_path = Path(graph_path) if graph_path else (
            SCORED_GRAPH_PATH if SCORED_GRAPH_PATH.exists() else FEATURED_GRAPH_PATH
        )
        if not selected_graph_path.exists():
            raise FileNotFoundError(f"Missing graph file: {selected_graph_path}")

        self.graph = ox.load_graphml(selected_graph_path)
        if "crs" not in self.graph.graph or self.graph.graph["crs"] == "epsg:4326":
            self.graph = ox.project_graph(self.graph)

        self.node_ids = list(self.graph.nodes())
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.idx_to_node = {idx: node_id for node_id, idx in self.node_to_idx.items()}
        self.out_edges = self._build_outgoing_edge_index()
        self.max_degree = max((len(edges) for edges in self.out_edges.values()), default=1)
        self.max_segment_time = max(
            (float(data.get("travel_time", 1.0)) for _, _, _, data in self.graph.edges(keys=True, data=True)),
            default=1.0,
        )
        self.node_xy = {
            node_id: (float(attrs["x"]), float(attrs["y"]))
            for node_id, attrs in self.graph.nodes(data=True)
        }
        self.candidate_nodes = [node for node, edges in self.out_edges.items() if edges]

        self.fixed_archetype_id = int(archetype_id)
        self.default_time_of_day = float(time_of_day)
        self.rng = np.random.default_rng(seed)

        high = np.array(
            [
                max(len(self.node_ids) - 1, 1),
                max(len(self.node_ids) - 1, 1),
                24.0,
                3.0,
                10000.0,
                100.0,
                100000.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=np.zeros(7, dtype=np.float32), high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_degree)

        self.origin_node: Any | None = None
        self.destination_node: Any | None = None
        self.current_node: Any | None = None
        self.archetype_id = self.fixed_archetype_id
        self.time_of_day = self.default_time_of_day
        self.steps_taken = 0
        self.max_steps = 1
        self.path_nodes: list[Any] = []
        self.path_edges: list[dict[str, Any]] = []
        self.cumulative_safety = 0.0

    def _build_outgoing_edge_index(self) -> dict[Any, list[tuple[Any, Any, dict[str, Any]]]]:
        out_edges: dict[Any, list[tuple[Any, Any, dict[str, Any]]]] = {}
        for node_id in self.graph.nodes():
            edges: list[tuple[Any, Any, dict[str, Any]]] = []
            for _, v, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                if "edge_id" not in data:
                    data["edge_id"] = f"{node_id}_{v}_{key}"
                if "safety_score" not in data:
                    data["safety_score"] = float(data.get("predicted_safety_score", 50.0))
                if "travel_time" not in data:
                    length = float(data.get("length", 1.0))
                    data["travel_time"] = max(length / 4.0, 0.1)
                edges.append((v, key, data))
            out_edges[node_id] = sorted(edges, key=lambda item: (str(item[0]), str(item[1])))
        return out_edges

    def _euclidean_distance(self, source_node: Any, target_node: Any) -> float:
        sx, sy = self.node_xy[source_node]
        tx, ty = self.node_xy[target_node]
        return float(math.hypot(tx - sx, ty - sy))

    def _shortest_path_length(self, source_node: Any, target_node: Any) -> int:
        try:
            path = nx.shortest_path(self.graph, source=source_node, target=target_node, weight="travel_time")
            return max(1, len(path) - 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 20

    def _sample_origin_destination(self) -> tuple[Any, Any]:
        for _ in range(200):
            origin = self.rng.choice(self.candidate_nodes)
            destination = self.rng.choice(self.candidate_nodes)
            if origin == destination:
                continue
            if self._euclidean_distance(origin, destination) < MIN_OD_DISTANCE_METERS:
                continue
            if not nx.has_path(self.graph, origin, destination):
                continue
            return origin, destination
        origin = self.candidate_nodes[0]
        destination = self.candidate_nodes[-1]
        return origin, destination

    def _get_observation(self) -> np.ndarray:
        if self.current_node is None or self.destination_node is None:
            raise RuntimeError("Environment state is not initialized")
        avg_safety = self.cumulative_safety / max(len(self.path_edges), 1)
        remaining_distance = self._euclidean_distance(self.current_node, self.destination_node)
        return np.array(
            [
                float(self.node_to_idx[self.current_node]),
                float(self.node_to_idx[self.destination_node]),
                float(self.time_of_day),
                float(self.archetype_id),
                float(self.steps_taken),
                float(avg_safety),
                float(remaining_distance),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        self.origin_node = options.get("origin_node")
        self.destination_node = options.get("destination_node")
        if self.origin_node is None or self.destination_node is None:
            self.origin_node, self.destination_node = self._sample_origin_destination()

        self.current_node = self.origin_node
        self.archetype_id = int(options.get("archetype_id", self.fixed_archetype_id))
        self.time_of_day = float(options.get("time_of_day", self.default_time_of_day))
        self.steps_taken = 0
        shortest_len = self._shortest_path_length(self.origin_node, self.destination_node)
        self.max_steps = int(max(3 * shortest_len, 10))
        self.path_nodes = [self.current_node]
        self.path_edges = []
        self.cumulative_safety = 0.0

        info = {
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "max_steps": self.max_steps,
            "valid_actions": len(self.out_edges.get(self.current_node, [])),
        }
        return self._get_observation(), info

    def compute_reward(self, edge: dict[str, Any], archetype_id: int, reached_goal: bool) -> float:
        weights = get_archetype_weights(archetype_id, hour_of_day=int(self.time_of_day))
        safety_gain = (float(edge.get("safety_score", 50.0)) / 100.0) * weights["beta"]
        time_penalty = -(float(edge.get("travel_time", self.max_segment_time)) / self.max_segment_time) * weights["alpha"]
        isolation_penalty = -float(edge.get("dead_end_penalty", 0.0)) * weights["vulnerability_weight"] * 0.3
        goal_bonus = 5.0 if reached_goal else 0.0
        step_penalty = -0.01
        return float(safety_gain + time_penalty + isolation_penalty + goal_bonus + step_penalty)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_node is None or self.destination_node is None:
            raise RuntimeError("Call reset() before step().")

        candidate_edges = self.out_edges.get(self.current_node, [])
        if not candidate_edges:
            obs = self._get_observation()
            return obs, -2.0, False, True, {"reason": "dead_end", "valid_actions": 0}

        invalid_action = int(action) >= len(candidate_edges)
        selected_index = 0 if invalid_action else int(action)
        next_node, _, edge_data = candidate_edges[selected_index]

        self.current_node = next_node
        self.steps_taken += 1
        self.path_nodes.append(next_node)
        self.path_edges.append(edge_data)
        self.cumulative_safety += float(edge_data.get("safety_score", 50.0))

        reached_goal = self.current_node == self.destination_node
        reward = self.compute_reward(edge_data, self.archetype_id, reached_goal)
        if invalid_action:
            reward -= 0.2

        terminated = bool(reached_goal)
        truncated = False
        info = {
            "edge_id": edge_data.get("edge_id"),
            "valid_actions": len(self.out_edges.get(self.current_node, [])),
            "invalid_action": invalid_action,
            "path_length": len(self.path_nodes),
            "mean_path_safety": self.cumulative_safety / max(len(self.path_edges), 1),
            "travel_time_sum": float(sum(float(edge.get("travel_time", 0.0)) for edge in self.path_edges)),
        }

        if self.steps_taken > self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
            info["reason"] = "max_steps_exceeded"

        return self._get_observation(), float(reward), bool(terminated), bool(truncated), info


def main() -> None:
    env = NightRouteEnv(archetype_id=ADAPTIVE, time_of_day=22.0)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=RANDOM_SEED)
    logger.info(f"Environment reset successful obs={obs.tolist()} info={info}")
    total_reward = 0.0
    for _ in range(10):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            logger.info(f"Episode ended terminated={terminated} truncated={truncated} info={step_info}")
            break
    logger.info(f"Sanity rollout complete total_reward={total_reward:.4f} final_obs={obs.tolist()}")


if __name__ == "__main__":
    main()

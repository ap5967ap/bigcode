from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from stable_baselines3 import PPO

from classifier.archetype_classifier import (
    ADAPTIVE,
    ARCHETYPE_NAMES,
    COMFORT_SEEKER,
    EFFICIENCY_FIRST,
    VULNERABLE_SOLO,
)
from rl.env import NightRouteEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ARCHETYPE_FILE_STEMS = {
    VULNERABLE_SOLO: "vulnerable_solo",
    COMFORT_SEEKER: "comfort_seeker",
    EFFICIENCY_FIRST: "efficiency_first",
    ADAPTIVE: "adaptive",
}


@dataclass
class AgentRunResult:
    path: list[Any]
    edges: list[dict[str, Any]]
    total_time: float
    mean_safety: float
    archetype_id: int
    archetype_name: str


class QLearningRouteAgent:
    def __init__(
        self,
        q_table: dict[str, list[float]],
        graph: nx.MultiDiGraph,
        archetype_id: int,
        max_degree: int,
    ) -> None:
        self.q_table = q_table
        self.graph = graph
        self.archetype_id = archetype_id
        self.max_degree = max_degree
        self.out_edges = self._build_outgoing_edge_index()

    @staticmethod
    def _state_key(node_id: Any, destination_node_id: Any, archetype_id: int) -> str:
        return f"{node_id}|{destination_node_id}|{archetype_id}"

    def _build_outgoing_edge_index(self) -> dict[Any, list[tuple[Any, Any, dict[str, Any]]]]:
        out_edges: dict[Any, list[tuple[Any, Any, dict[str, Any]]]] = {}
        for node_id in self.graph.nodes():
            edges: list[tuple[Any, Any, dict[str, Any]]] = []
            for _, v, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                if "edge_id" not in data:
                    data["edge_id"] = f"{node_id}_{v}_{key}"
                edges.append((v, key, data))
            out_edges[node_id] = sorted(edges, key=lambda item: (str(item[0]), str(item[1])))
        return out_edges

    @classmethod
    def load(cls, path: Path, graph: nx.MultiDiGraph) -> "QLearningRouteAgent":
        with zipfile.ZipFile(path, "r") as archive:
            payload = json.loads(archive.read("agent.json").decode("utf-8"))
        return cls(
            q_table=payload["q_table"],
            graph=graph,
            archetype_id=int(payload["archetype_id"]),
            max_degree=int(payload["max_degree"]),
        )

    @staticmethod
    def save(path: Path, q_table: dict[str, list[float]], archetype_id: int, max_degree: int) -> None:
        payload = {
            "algorithm": "q_learning",
            "archetype_id": archetype_id,
            "max_degree": max_degree,
            "q_table": q_table,
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("agent.json", json.dumps(payload))

    def predict(self, node_id: Any, destination_node_id: Any) -> int:
        candidates = self.out_edges.get(node_id, [])
        if not candidates:
            return 0
        state_key = self._state_key(node_id, destination_node_id, self.archetype_id)
        values = np.array(self.q_table.get(state_key, [0.0] * self.max_degree), dtype=np.float32)
        valid_values = values[: len(candidates)]
        return int(np.argmax(valid_values))

    def rollout(self, origin_node: Any, destination_node: Any, max_steps: int = 500) -> AgentRunResult:
        path = [origin_node]
        edges: list[dict[str, Any]] = []
        current = origin_node
        for _ in range(max_steps):
            candidates = self.out_edges.get(current, [])
            if not candidates:
                break
            action = self.predict(current, destination_node)
            next_node, _, edge_data = candidates[min(action, len(candidates) - 1)]
            edges.append(edge_data)
            path.append(next_node)
            current = next_node
            if current == destination_node:
                break

        total_time = float(sum(float(edge.get("travel_time", 0.0)) for edge in edges))
        mean_safety = float(np.mean([float(edge.get("safety_score", 50.0)) for edge in edges])) if edges else 0.0
        return AgentRunResult(
            path=path,
            edges=edges,
            total_time=total_time,
            mean_safety=mean_safety,
            archetype_id=self.archetype_id,
            archetype_name=ARCHETYPE_NAMES[self.archetype_id],
        )


class PPOPolicyRouteAgent:
    def __init__(self, model: PPO, archetype_id: int) -> None:
        self.model = model
        self.archetype_id = archetype_id

    @classmethod
    def load(cls, path: Path, archetype_id: int, env: NightRouteEnv) -> "PPOPolicyRouteAgent":
        return cls(PPO.load(path, env=env), archetype_id=archetype_id)

    def rollout(
        self,
        env: NightRouteEnv,
        origin_node: Any,
        destination_node: Any,
        time_of_day: float = 21.0,
    ) -> AgentRunResult:
        obs, _ = env.reset(
            options={
                "origin_node": origin_node,
                "destination_node": destination_node,
                "archetype_id": self.archetype_id,
                "time_of_day": time_of_day,
            }
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
        total_time = float(sum(float(edge.get("travel_time", 0.0)) for edge in env.path_edges))
        mean_safety = float(np.mean([float(edge.get("safety_score", 50.0)) for edge in env.path_edges])) if env.path_edges else 0.0
        return AgentRunResult(
            path=list(env.path_nodes),
            edges=list(env.path_edges),
            total_time=total_time,
            mean_safety=mean_safety,
            archetype_id=self.archetype_id,
            archetype_name=ARCHETYPE_NAMES[self.archetype_id],
        )


def agent_path_for_archetype(archetype_id: int) -> Path:
    return PROCESSED_DIR / f"agent_{ARCHETYPE_FILE_STEMS[archetype_id]}.zip"

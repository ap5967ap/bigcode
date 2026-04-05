from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame
from loguru import logger
from shapely.geometry import LineString


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GRAPH_PATH = RAW_DIR / "road_graph.graphml"
POI_PATH = RAW_DIR / "pois.geojson"

BENGALURU_BBOX = (12.995, 12.950, 77.640, 77.590)  # north, south, east, west
CHENNAI_BBOX = (13.110, 13.065, 80.275, 80.225)
POI_TAGS = {
    "amenity": ["restaurant", "cafe", "hospital", "pharmacy", "police", "fuel", "bank", "school"],
    "shop": True,
    "public_transport": ["stop_position", "platform"],
}
HIGHWAY_SPEED_KMPH = {
    "motorway": 55.0,
    "trunk": 45.0,
    "primary": 40.0,
    "secondary": 30.0,
    "tertiary": 25.0,
    "residential": 18.0,
    "service": 12.0,
    "living_street": 10.0,
    "unclassified": 15.0,
    "path": 5.0,
    "track": 8.0,
    "footway": 5.0,
    "cycleway": 12.0,
    "steps": 3.0,
}
DEFAULT_SPEED_KMPH = 15.0


def _normalize_highway_type(highway: Any) -> str:
    if isinstance(highway, list) and highway:
        return str(highway[0])
    if highway is None:
        return "unclassified"
    return str(highway)


def _edge_geometry(graph: nx.MultiDiGraph, u: int, v: int, data: dict[str, Any]) -> LineString:
    if "geometry" in data and isinstance(data["geometry"], LineString):
        return data["geometry"]
    ux = float(graph.nodes[u]["x"])
    uy = float(graph.nodes[u]["y"])
    vx = float(graph.nodes[v]["x"])
    vy = float(graph.nodes[v]["y"])
    return LineString([(ux, uy), (vx, vy)])


def _travel_time_seconds(length_m: float, highway_type: str) -> float:
    speed_kmph = HIGHWAY_SPEED_KMPH.get(highway_type, DEFAULT_SPEED_KMPH)
    speed_mps = max(speed_kmph * 1000.0 / 3600.0, 0.5)
    return float(length_m / speed_mps)


def _download_graph_with_retry(bbox: tuple[float, float, float, float], city_name: str, retries: int = 3) -> nx.MultiDiGraph:
    north, south, east, west = bbox
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading OSM road graph for {city_name}, attempt {attempt}/{retries}")
            return ox.graph_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                network_type="all",
                simplify=True,
                retain_all=False,
                truncate_by_edge=True,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(f"Graph download attempt {attempt} failed for {city_name}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download OSM graph for {city_name}") from last_error


def _download_pois_with_retry(bbox: tuple[float, float, float, float], city_name: str, retries: int = 3) -> GeoDataFrame:
    north, south, east, west = bbox
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading OSM POIs for {city_name}, attempt {attempt}/{retries}")
            pois = ox.features_from_bbox(north=north, south=south, east=east, west=west, tags=POI_TAGS)
            if pois.empty:
                logger.warning(f"No POIs returned for {city_name}; continuing with empty GeoDataFrame")
            return pois.reset_index()
        except Exception as exc:
            last_error = exc
            logger.warning(f"POI download attempt {attempt} failed for {city_name}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download OSM POIs for {city_name}") from last_error


def build_road_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    for u, v, key, data in graph.edges(keys=True, data=True):
        highway_type = _normalize_highway_type(data.get("highway"))
        length_m = float(data.get("length", 0.0))
        geometry = _edge_geometry(graph, u, v, data)
        data["highway"] = highway_type
        data["length"] = length_m
        data["travel_time"] = _travel_time_seconds(length_m, highway_type)
        data["geometry"] = geometry
        data["edge_id"] = f"{u}_{v}_{key}"
    return graph


def download_city_zone() -> tuple[nx.MultiDiGraph, GeoDataFrame, str]:
    try:
        graph = _download_graph_with_retry(BENGALURU_BBOX, "Bengaluru")
        pois = _download_pois_with_retry(BENGALURU_BBOX, "Bengaluru")
        return graph, pois, "Bengaluru"
    except Exception as exc:
        logger.warning(f"Bengaluru download failed, falling back to Chennai: {exc}")
        graph = _download_graph_with_retry(CHENNAI_BBOX, "Chennai")
        pois = _download_pois_with_retry(CHENNAI_BBOX, "Chennai")
        return graph, pois, "Chennai"


def save_artifacts(graph: nx.MultiDiGraph, pois: GeoDataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=GRAPH_PATH)
    if pois.empty:
        GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326").to_file(POI_PATH, driver="GeoJSON")
    else:
        pois.to_file(POI_PATH, driver="GeoJSON")
    logger.info(f"Saved graph to {GRAPH_PATH}")
    logger.info(f"Saved POIs to {POI_PATH}")


def main() -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    graph, pois, city_name = download_city_zone()
    graph = build_road_graph(graph)
    save_artifacts(graph, pois)
    logger.info(
        f"Completed OSM load for {city_name}: nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, pois={len(pois)}"
    )


if __name__ == "__main__":
    main()

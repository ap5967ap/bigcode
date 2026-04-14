import networkx as nx

from routing.router import NightSafeRouter


def test_router_returns_route_result_payload() -> None:
    router = NightSafeRouter()
    node_ids = list(router.graph.nodes())
    origin_node = node_ids[100]
    destination_node = next(node for node in node_ids[500:] if nx.has_path(router.graph, origin_node, node))
    result = router.route(
        origin_coords=router._node_to_latlon(origin_node),
        destination_coords=router._node_to_latlon(destination_node),
        user_context={
            "travel_mode": "walking",
            "hour_of_day": 22,
            "is_female": True,
            "destination_type": "residential",
            "query_day_type": 0,
        },
    )

    assert set(result) == {
        "fastest",
        "balanced",
        "safest",
        "agent_route",
        "pareto_frontier",
        "segment_explanations",
        "service_area_bounds",
    }
    assert len(result["fastest"]["edges"]) > 0
    assert len(result["pareto_frontier"]) == 7
    assert result["agent_route"]["archetype"]
    assert result["fastest"]["total_distance"] > 0
    assert result["balanced"]["total_distance"] > 0
    assert result["safest"]["total_distance"] > 0
    assert result["fastest"]["total_time"] <= result["balanced"]["total_time"]
    assert result["fastest"]["total_time"] <= result["safest"]["total_time"]
    assert result["safest"]["mean_safety"] >= result["balanced"]["mean_safety"]
    assert result["safest"]["mean_safety"] >= result["fastest"]["mean_safety"]
    first_edge = result["fastest"]["edges"][0]
    first_lon, first_lat = first_edge["geometry"]["coordinates"][0]
    assert -180.0 <= first_lon <= 180.0
    assert -90.0 <= first_lat <= 90.0
    assert result["service_area_bounds"]["min_lat"] < result["service_area_bounds"]["max_lat"]


def test_router_routes_change_for_different_user_contexts() -> None:
    router = NightSafeRouter()
    origin = [12.9716, 77.5946]
    destination = [12.985, 77.61]

    night_female_walk = router.route(
        origin_coords=origin,
        destination_coords=destination,
        user_context={
            "travel_mode": "walking",
            "hour_of_day": 2,
            "is_female": True,
            "destination_type": "residential",
            "query_day_type": 0,
        },
    )
    day_male_cab = router.route(
        origin_coords=origin,
        destination_coords=destination,
        user_context={
            "travel_mode": "cab",
            "hour_of_day": 15,
            "is_female": False,
            "destination_type": "residential",
            "query_day_type": 0,
        },
    )

    route_keys = ["fastest", "balanced", "safest", "agent_route"]
    changed_routes = 0
    for route_key in route_keys:
        first = night_female_walk[route_key]
        second = day_male_cab[route_key]
        if (
            abs(first["total_time"] - second["total_time"]) > 1e-6
            or abs(first["mean_safety"] - second["mean_safety"]) > 1e-6
            or first["path"] != second["path"]
        ):
            changed_routes += 1

    assert changed_routes >= 1

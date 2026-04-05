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
    assert len(result["pareto_frontier"]) == 5
    assert result["agent_route"]["archetype"]
    first_edge = result["fastest"]["edges"][0]
    first_lon, first_lat = first_edge["geometry"]["coordinates"][0]
    assert -180.0 <= first_lon <= 180.0
    assert -90.0 <= first_lat <= 90.0
    assert result["service_area_bounds"]["min_lat"] < result["service_area_bounds"]["max_lat"]

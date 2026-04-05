from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint_reports_graph_stats() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["graph_nodes"] > 0
        assert payload["graph_edges"] > 0
        assert payload["service_area_bounds"]["min_lat"] < payload["service_area_bounds"]["max_lat"]


def test_route_rejects_destination_outside_service_area() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/route",
            json={
                "origin": [12.9716, 77.5946],
                "destination": [28.6139, 77.2090],
                "travel_mode": "walking",
                "hour_of_day": 22,
                "is_female": False,
                "destination_type": "residential",
            },
        )

    assert response.status_code == 400
    assert "outside the supported Bangalore service area" in response.json()["detail"]

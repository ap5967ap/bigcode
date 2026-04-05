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

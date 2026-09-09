"""Integration tests: HTTP requests through the real FastAPI app.

These exercise the full path (routing, validation, model inference,
serialisation) rather than a single unit.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_schema_describes_every_form_field(client: TestClient) -> None:
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["total_model_columns"] == 79
    names = [field["name"] for field in body["fields"]]
    assert "Neighborhood" in names
    assert "1stFlrSF" in names

    neighborhood = next(f for f in body["fields"] if f["name"] == "Neighborhood")
    assert neighborhood["type"] == "select"
    assert len(neighborhood["options"]) > 1

    living_area = next(f for f in body["fields"] if f["name"] == "GrLivArea")
    assert living_area["type"] == "number"
    assert living_area["minimum"] < living_area["maximum"]


def test_predict_returns_200_and_a_price(client: TestClient, valid_payload: dict) -> None:
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "USD"
    assert isinstance(body["sale_price"], float)
    assert 20_000 < body["sale_price"] < 2_000_000


def test_predict_rejects_invalid_input_with_422(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload) | {"OverallQual": 99}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field_with_422(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    del payload["Neighborhood"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_neighborhood_with_422(
    client: TestClient, valid_payload: dict
) -> None:
    payload = dict(valid_payload) | {"Neighborhood": "Atlantis"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_openapi_document_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]

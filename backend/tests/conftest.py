"""Shared pytest fixtures for the backend test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="session")
def valid_payload() -> dict:
    """A realistic, mid-range house that every test can reuse."""
    return {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "YearBuilt": 2003,
        "GarageCars": 2,
        "FullBath": 2,
        "TotRmsAbvGrd": 8,
        "LotArea": 8450,
        "Neighborhood": "CollgCr",
    }


@pytest.fixture(scope="session")
def client() -> TestClient:
    """TestClient with lifespan enabled so the model is loaded once."""
    with TestClient(app) as test_client:
        yield test_client

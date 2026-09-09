"""Tests for the static frontend.

The UI is plain HTML/CSS/JS with no build step, so these tests assert on the
delivered source: that the markup carries the hooks app.js looks up, that the
client talks to the endpoints the backend actually exposes, and that the
container wiring is consistent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parent.parent

INDEX_HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
CONFIG_JS = (FRONTEND_DIR / "config.js").read_text(encoding="utf-8")

#: Element ids app.js resolves with getElementById at start-up.
REQUIRED_IDS = [
    "api-status",
    "fields",
    "predict-form",
    "submit-button",
    "reset-button",
    "result",
    "result-price",
    "result-meta",
    "error",
    "error-detail",
]


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_markup_contains_every_id_the_script_needs(element_id: str) -> None:
    assert f'id="{element_id}"' in INDEX_HTML


def test_index_loads_config_before_app() -> None:
    assert INDEX_HTML.index("config.js") < INDEX_HTML.index("app.js")


def test_index_links_the_stylesheet() -> None:
    assert 'href="style.css"' in INDEX_HTML
    assert (FRONTEND_DIR / "style.css").exists()


def test_script_calls_the_documented_endpoints() -> None:
    assert '"/health"' in APP_JS or "/health" in APP_JS
    assert "/schema" in APP_JS
    assert "/predict" in APP_JS


def test_predict_is_posted_as_json() -> None:
    assert 'method: "POST"' in APP_JS
    assert '"Content-Type": "application/json"' in APP_JS


def test_form_is_built_from_the_backend_schema_not_hard_coded() -> None:
    # Field names must come from the /schema response, so the source should not
    # contain a hard-coded list of Kaggle column names.
    assert "GrLivArea" not in APP_JS
    assert "renderForm" in APP_JS


def test_config_defines_an_api_url() -> None:
    assert re.search(r"window\.API_URL\s*=", CONFIG_JS)


def test_entrypoint_overwrites_config_with_the_api_url_env_var() -> None:
    entrypoint = (FRONTEND_DIR / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "API_URL" in entrypoint
    assert "config.js" in entrypoint


def test_nginx_serves_a_health_route_used_by_the_healthcheck() -> None:
    nginx_conf = (FRONTEND_DIR / "nginx.conf").read_text(encoding="utf-8")
    assert "/healthz" in nginx_conf

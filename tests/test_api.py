import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import app
from conftest import make_classification_df, make_messy_df

client = TestClient(app)


def upload(df, target=None, filename="data.csv"):
    buffer = io.BytesIO(df.to_csv(index=False).encode())
    data = {"target": target} if target else {}
    return client.post("/api/prepare", files={"file": (filename, buffer, "text/csv")}, data=data)


def test_health_and_samples():
    assert client.get("/api/health").json() == {"status": "ok"}
    ids = {entry["id"] for entry in client.get("/api/samples").json()}
    assert {"fashion", "retail", "titanic", "freelancers", "penguins"} == ids


def test_prepare_returns_a_complete_serialisable_result():
    response = upload(make_classification_df(), target="segment")

    assert response.status_code == 200
    result = response.json()
    assert json.loads(json.dumps(result)) == result
    assert result["target"] == "segment" and result["task_type"] == "classification"
    assert [s["id"] for s in result["steps"]] == ["load", "clean", "duplicates", "impute", "encode", "scale", "report"]
    assert 0 <= result["stats"]["quality_before"] <= result["stats"]["quality_after"] <= 1
    assert result["summary"]["en"] and result["summary"]["de"] and result["summary"]["kind"] == "automatic"
    assert all(card["title"]["en"] and card["title"]["de"] for card in result["decisions"])
    assert "customer_id" in result["preview"]["columns"]
    assert pd.read_csv(io.StringIO(result["downloads"]["clean_csv"])).shape[0] == result["dataset"]["rows_out"]


def test_every_highlighted_cell_has_an_explanation_in_both_languages():
    result = upload(make_messy_df()).json()

    reasons = result["preview"]["reasons"]
    for row in result["preview"]["rows"]:
        for cell in row["cells"]:
            if cell["kind"] in ("fixed", "removed", "note"):
                assert cell["reason"] in reasons
                assert reasons[cell["reason"]]["en"] and reasons[cell["reason"]]["de"]


def test_fixed_cells_show_before_and_after():
    result = upload(make_messy_df()).json()
    columns = result["preview"]["columns"]
    rate = columns.index("hourly_rate")

    cells = [row["cells"][rate] for row in result["preview"]["rows"]]
    fixed = [c for c in cells if c["kind"] == "fixed"]
    assert fixed and all(c["before"] != c["after"] for c in fixed)
    assert any(c["before"].startswith(("$", "USD")) for c in fixed)


def test_target_options_are_offered_for_uploads():
    result = upload(make_classification_df()).json()

    options = result["dataset"]["target_options"]
    assert "segment" in options and "region" in options
    assert "customer_id" not in options and "name" not in options


def test_unknown_target_is_a_clear_400():
    response = upload(make_classification_df(), target="does_not_exist")
    assert response.status_code == 400
    assert "does_not_exist" in response.json()["detail"]


def test_wrong_file_type_and_size_limits(monkeypatch):
    assert upload(make_classification_df(), filename="data.txt").status_code == 400
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 100)
    assert upload(make_classification_df()).status_code == 413


def test_row_limit(monkeypatch):
    monkeypatch.setattr(api.main, "MAX_ROWS", 50)
    assert upload(make_classification_df()).status_code == 413


def test_garbage_file_is_rejected_politely():
    response = client.post("/api/prepare", files={"file": ("x.csv", io.BytesIO(b"\x00\x01\x02 not a table"), "text/csv")})
    assert response.status_code in (400, 422)
    assert "Traceback" not in response.text

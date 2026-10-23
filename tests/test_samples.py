import json
import os

import pandas as pd
import pytest

from samples.build_samples import build_fashion_catalog

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_fashion_catalog_is_reproducible_and_messy():
    df = build_fashion_catalog()
    assert df.equals(build_fashion_catalog())

    assert len(df) == 1200 and df.duplicated().sum() == 6
    assert df["price"].str.contains("€").any() and df["price"].str.startswith("EUR").any()
    assert df["colour"].str.isupper().any() and df["colour"].str.endswith(" ").any()
    assert {"Ja", "nein"} <= set(df["on_sale"])
    assert df["stock"].isna().any()
    assert df["launch_date"].str.fullmatch(r"\d{2}\.\d{2}\.\d{4}").all()


def test_catalog_entries_point_to_existing_files():
    with open(os.path.join(REPO_DIR, "samples", "catalog.json")) as f:
        catalog = json.load(f)

    assert {entry["id"] for entry in catalog} == {"fashion", "retail", "titanic", "freelancers", "penguins"}
    for entry in catalog:
        df = pd.read_csv(os.path.join(REPO_DIR, entry["file"]))
        assert (len(df), df.shape[1]) == (entry["rows"], entry["columns"])
        assert set(entry["targets"]) <= set(df.columns)
        assert entry["name"]["en"] and entry["name"]["de"]
        assert entry["licence"]


@pytest.mark.parametrize("sample_id", ["fashion", "retail", "titanic", "penguins"])
def test_every_sample_prepares_cleanly(run_pipeline, sample_id):
    with open(os.path.join(REPO_DIR, "samples", "catalog.json")) as f:
        entry = next(e for e in json.load(f) if e["id"] == sample_id)
    df = pd.read_csv(os.path.join(REPO_DIR, entry["file"]))

    result = run_pipeline(df, entry["targets"][0])

    assert result.processed.isnull().sum().sum() == 0
    assert not [d for d in result.decisions("Evaluation") if "(Failed)" in d["decision"]]

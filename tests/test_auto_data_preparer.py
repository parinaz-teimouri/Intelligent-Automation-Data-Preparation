import csv
import math

import numpy as np
import pandas as pd
import pytest

from conftest import DATASETS
from data_preparation import AutoDataPreparer, make_imputer, make_scaler


@pytest.mark.parametrize("name", DATASETS)
def test_ml_output_is_complete_and_numeric(run_pipeline, name):
    make_df, target = DATASETS[name]
    result = run_pipeline(make_df(), target)

    assert result.processed.isnull().sum().sum() == 0
    non_numeric = result.processed.select_dtypes(exclude=[np.number]).columns
    assert list(non_numeric) == []


# --- Target handling -------------------------------------------------------------

def test_rows_with_missing_target_are_dropped(run_pipeline):
    df = DATASETS["classification"][0]()
    result = run_pipeline(df, "segment")

    assert len(result.processed) == df["segment"].notna().sum()


def test_classification_target_is_label_encoded_not_transformed(run_pipeline):
    df = DATASETS["classification"][0]()
    result = run_pipeline(df, "segment")

    target = result.processed["segment"]
    assert pd.api.types.is_integer_dtype(target)
    assert set(target.unique()) == {0, 1, 2}
    # Class balance survives encoding: same counts per class as the raw labels.
    assert sorted(target.value_counts()) == sorted(df["segment"].value_counts())


def test_regression_target_values_are_unchanged(run_pipeline):
    df = DATASETS["regression"][0]()
    result = run_pipeline(df, "price")

    expected = np.sort(df["price"].dropna().to_numpy())
    actual = np.sort(result.processed["price"].to_numpy())
    np.testing.assert_allclose(actual, expected)  # no imputing, capping or scaling
    assert actual.max() == pytest.approx(500.0)


def test_float_rating_target_is_detected_as_regression(run_pipeline):
    result = run_pipeline(DATASETS["rating"][0](), "rating")

    assert result.decisions("Task Detection")[0]["decision"] == "Regression"


# --- Performance-driven evaluation -----------------------------------------------

@pytest.mark.parametrize("name", ["classification", "regression", "rating"])
def test_candidate_strategies_are_actually_compared(run_pipeline, name):
    make_df, target = DATASETS[name]
    result = run_pipeline(make_df(), target)

    for step in ["Imputation", "Scaling"]:
        details = result.winner(step)["details"]
        assert len(details["candidates"]) >= 2, f"{step}: nothing to compare"
        assert math.isfinite(details["best_score"])
        assert details["best_score"] > -1, f"{step}: evaluation failed"


def test_failed_evaluations_are_reported_and_not_stored_as_feedback(run_pipeline, monkeypatch):
    def broken_pipeline(self, *args, **kwargs):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(AutoDataPreparer, "_build_eval_pipeline", broken_pipeline)

    result = run_pipeline(DATASETS["classification"][0](), "segment")

    failures = [d for d in result.decisions("Evaluation") if "(Failed)" in d["decision"]]
    assert failures and "model exploded" in failures[0]["reason"]
    for step in ["Imputation", "Scaling"]:
        assert any("(Fallback)" in d["decision"] for d in result.decisions(step))
        assert not any("Winner" in d["decision"] for d in result.decisions(step))
    assert not list((result.workdir / "meta_models").glob("*_feedback.csv"))


@pytest.mark.parametrize("factory", [make_imputer, make_scaler])
def test_unknown_strategy_names_raise(factory):
    with pytest.raises(ValueError):
        factory("not-a-strategy")


def test_target_does_not_leak_into_evaluation_features(run_pipeline):
    # The features are pure noise, so an honest evaluation scores near chance
    # (macro F1 ~0.33). A score near 1.0 means the target was used as a feature.
    result = run_pipeline(DATASETS["noise"][0](), "label")

    assert result.winner("Scaling")["details"]["best_score"] < 0.6


# --- Data cleaning ---------------------------------------------------------------

def test_identifier_and_free_text_columns_excluded_from_ml_output(run_pipeline):
    result = run_pipeline(DATASETS["classification"][0](), "segment")

    assert "customer_id" not in result.processed.columns
    assert "name" not in result.processed.columns
    # ...but kept in the human-readable file for traceability.
    assert {"customer_id", "name"} <= set(result.human_readable.columns)


def test_numeric_strings_are_parsed(run_pipeline):
    result = run_pipeline(DATASETS["messy"][0]())
    hr = result.human_readable

    assert pd.api.types.is_numeric_dtype(hr["hourly_rate"])
    assert set(hr["hourly_rate"].unique()) <= {30, 50, 75, 100}
    assert pd.api.types.is_numeric_dtype(hr["satisfaction"])
    assert hr["satisfaction"].between(40, 100).all()


def test_boolean_strings_become_zero_one(run_pipeline):
    result = run_pipeline(DATASETS["messy"][0]())

    assert set(result.human_readable["is_active"].unique()) <= {0, 1}


def test_category_spellings_are_merged(run_pipeline):
    result = run_pipeline(DATASETS["messy"][0]())

    assert result.human_readable["gender"].nunique() == 2


# --- Meta-learning feedback ------------------------------------------------------

def test_feedback_files_are_well_formed(run_pipeline):
    result = run_pipeline(DATASETS["classification"][0](), "segment")

    feedback_files = list((result.workdir / "meta_models").glob("*_feedback.csv"))
    assert feedback_files, "no feedback stored"
    for path in feedback_files:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        header, body = rows[0], rows[1:]
        assert "best_strategy" in header
        assert body and all(len(row) == len(header) for row in body)

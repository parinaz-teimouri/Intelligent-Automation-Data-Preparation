import json
import os

import numpy as np
import pandas as pd
import pytest

from data_preparation import AutoDataPreparer


# --- Synthetic datasets -------------------------------------------------------
# Small, seeded datasets that each exercise one behaviour of the pipeline.

def make_classification_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "name": [f"Person {i}" for i in range(n)],
        "age": rng.integers(18, 70, size=n).astype(float),
        "income": rng.lognormal(10, 0.5, size=n),
        "score": score,
        "region": rng.choice(["north", "south", "east", "west"], size=n),
        "plan": rng.choice(["basic", "pro"], size=n),
        "segment": np.where(score > 0.5, "high", np.where(score < -0.5, "low", "mid")),
    })
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "region"] = np.nan
    df.loc[[3, 40, 77, 101, 140], "segment"] = np.nan
    return df


def make_regression_df(n=150, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    price = 3 * x1 + 2 * x2 + rng.normal(scale=0.5, size=n) + 50
    price[10] = 500.0  # a rare extreme target value that must not be capped
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "colour": rng.choice(["red", "green", "blue"], size=n),
        "price": price,
    })
    df.loc[rng.choice(n, 10, replace=False), "x1"] = np.nan
    df.loc[[5, 60, 120], "price"] = np.nan
    return df


def make_rating_df(n=1200, seed=2):
    """Float target with one decimal (like a 0-5 star rating): a regression task."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    rating = np.clip(np.round(2.5 + x + rng.normal(scale=0.3, size=n), 1), 0, 5)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "rating": rating})
    df.loc[rng.choice(n, 50, replace=False), "x"] = np.nan
    return df


def make_noise_df(n=150, seed=3):
    """Features carry no information about the integer target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    df["label"] = rng.integers(0, 3, size=n)
    df.loc[rng.choice(n, 10, replace=False), "f0"] = np.nan
    return df


def make_messy_df(n=120, seed=4):
    rng = np.random.default_rng(seed)
    rates = rng.choice([30, 50, 75, 100], size=n)
    rate_fmt = rng.choice(["{}", "${}", "USD {}"], size=n)
    df = pd.DataFrame({
        "hourly_rate": [f.format(r) for f, r in zip(rate_fmt, rates)],
        "satisfaction": [f"{v}%" for v in rng.integers(40, 100, size=n)],
        "is_active": rng.choice(["Y", "N", "True", "False", "1", "0"], size=n),
        "gender": rng.choice(["m", "M", "Male", "MALE", "male", "f", "F", "Female", "FEMALE", "female"], size=n),
        "experience": rng.integers(0, 30, size=n).astype(float),
    })
    df.loc[rng.choice(n, 10, replace=False), "satisfaction"] = np.nan
    return df


DATASETS = {
    "classification": (make_classification_df, "segment"),
    "regression": (make_regression_df, "price"),
    "rating": (make_rating_df, "rating"),
    "noise": (make_noise_df, "label"),
    "messy": (make_messy_df, None),
}


# --- Pipeline runner ------------------------------------------------------------

class PipelineResult:
    def __init__(self, raw, processed, human_readable, report, workdir, chosen_strategies):
        self.raw = raw
        self.processed = processed
        self.human_readable = human_readable
        self.report = report
        self.workdir = workdir
        self.chosen_strategies = chosen_strategies

    def decisions(self, step):
        return [d for d in self.report["decisions"] if d["step"] == step]

    def winner(self, step):
        """The 'Winner' decision logged for an optimised step (Imputation / Scaling)."""
        winners = [d for d in self.decisions(step) if "Winner" in d["decision"]]
        assert winners, f"no evaluated winner logged for {step}"
        return winners[0]


@pytest.fixture
def run_pipeline(tmp_path):
    """Runs the full pipeline on a DataFrame inside an isolated directory."""

    def _run(df, target_col=None, **options):
        input_path = tmp_path / "input.csv"
        df.to_csv(input_path, index=False)
        output_path = tmp_path / "processed_input.csv"

        # Isolate meta-models and feedback files from the repository's meta_models/ folder.
        preparer = AutoDataPreparer(str(input_path), target_col=target_col,
                                    models_dir=str(tmp_path / "meta_models"), **options)
        preparer.run_pipeline(str(output_path))

        with open(tmp_path / "report_processed_input.json") as f:
            report = json.load(f)
        return PipelineResult(
            raw=df,
            processed=pd.read_csv(output_path),
            human_readable=pd.read_csv(tmp_path / "human_readable_input.csv"),
            report=report,
            workdir=tmp_path,
            chosen_strategies=preparer.chosen_strategies,
        )

    return _run

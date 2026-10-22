import math

import numpy as np
import pandas as pd
import pytest

import benchmark
from conftest import DATASETS

STATIC = {"imputation": "mean", "scaling": "standard", "encoding": "onehot"}


# --- AutoDataPreparer options ------------------------------------------------------

def test_fixed_strategies_skip_selection(run_pipeline):
    result = run_pipeline(DATASETS["classification"][0](), "segment", fixed_strategies=STATIC)

    assert not result.decisions("Evaluation")
    assert any("(Fixed)" in d["decision"] for d in result.decisions("Imputation"))
    assert result.chosen_strategies["imputation"] == "mean"
    assert result.chosen_strategies["scaling"] == "standard"
    assert set(result.chosen_strategies["encoding"].values()) == {"onehot"}


def test_optimize_false_only_predicts(run_pipeline):
    result = run_pipeline(DATASETS["classification"][0](), "segment", optimize=False)

    assert not result.decisions("Evaluation")
    assert any(d["reason"] == "Predicted by Meta-Model" for d in result.decisions("Scaling"))


def test_standardize_false_leaves_values_and_identifiers(run_pipeline):
    result = run_pipeline(DATASETS["messy"][0](), standardize=False)

    assert not result.decisions("Standardization")
    assert result.human_readable["gender"].nunique() == 10


def test_record_feedback_false_stores_nothing(run_pipeline):
    result = run_pipeline(DATASETS["classification"][0](), "segment", record_feedback=False)

    assert result.winner("Imputation")  # evaluation still ran
    assert not list((result.workdir / "meta_models").glob("*_feedback.csv"))


# --- Benchmark -----------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    from sklearn.datasets import load_wine
    return benchmark._sklearn_dataset(load_wine, "cultivar", "classification", "wine (defects)")


def test_injected_defects_are_present_and_target_untouched(small_dataset):
    name, df, target, task_type, _ = small_dataset

    assert df["record_id"].is_unique
    assert df.isnull().sum().sum() > 0
    assert any(df[c].astype(str).str.startswith("$").any() for c in df.columns)
    level_col = next(c for c in df.columns if c.endswith("_level"))
    assert df[level_col].nunique() > 3  # several spellings of low/medium/high
    assert df[target].notna().all() and df[target].nunique() == 3


@pytest.mark.parametrize("config", benchmark.CONFIGS)
def test_every_configuration_produces_scorable_output(small_dataset, tmp_path, config):
    _, df, target, task_type, _ = small_dataset
    selection, evaluation = df.iloc[::2], df.iloc[1::2]

    processed, chosen = benchmark.run_config(config, selection, evaluation, target, str(tmp_path),
                                             models_dir=str(tmp_path / "models"))
    scores = benchmark.downstream_scores(processed, target, task_type, seed=0)

    assert processed.isnull().sum().sum() == 0
    assert set(scores) == {"linear", "random forest"}
    assert all(math.isfinite(s) for s in scores.values())
    if config == "full system":
        assert {"imputation", "scaling"} <= set(chosen)


def test_full_system_applies_choices_made_on_the_selection_half(small_dataset, tmp_path, monkeypatch):
    _, df, target, _, _ = small_dataset
    calls = []
    original_prepare = benchmark.prepare

    def recording_prepare(data, *args, **options):
        calls.append((len(data), options))
        return original_prepare(data, *args, **options)
    monkeypatch.setattr(benchmark, "prepare", recording_prepare)

    selection, evaluation = df.iloc[:60], df.iloc[60:]
    benchmark.run_config("full system", selection, evaluation, target, str(tmp_path), str(tmp_path / "models"))

    (select_rows, select_opts), (eval_rows, eval_opts) = calls
    assert select_rows == 60 and select_opts["optimize"] is True
    assert eval_rows == len(df) - 60 and eval_opts["optimize"] is False
    assert {"imputation", "scaling"} <= set(eval_opts["fixed_strategies"])


def test_markdown_report_marks_best_and_ranks():
    rows = []
    for dataset, task in [("a", "classification"), ("b", "regression")]:
        for config, base in zip(benchmark.CONFIGS, [0.5, 0.6, 0.7, 0.8]):
            for model in ["linear", "random forest"]:
                for repeat in range(2):
                    rows.append({"dataset": dataset, "task": task, "repeat": repeat, "config": config, "model": model,
                                 "score": base + 0.01 * repeat, "prep_seconds": 0.1, "features": 5,
                                 "imputation": "mean", "scaling": "standard"})
    tables = benchmark.summary_tables(pd.DataFrame(rows))
    markdown = benchmark.to_markdown(*tables, datasets=[], repeats=2, meta_status="fixed rules")

    assert "**0.805 (0.007)**" in markdown  # full system is best in each row
    full_rank = tables[2]["full system"]
    assert full_rank == 1.0
    pairs = tables[-1].set_index("comparison")
    assert pairs.loc["full system vs static", "wins"] == 4  # 2 datasets x 2 repeats
    assert pairs.loc["full system vs static", "mean_difference"] == pytest.approx(0.3)
    assert "## Head to head" in markdown

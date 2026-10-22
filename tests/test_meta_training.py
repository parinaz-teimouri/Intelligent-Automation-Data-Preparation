import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import train_meta_model
from meta_learning import FEATURE_NAMES, FEEDBACK_COLUMNS, STRATEGIES, MetaLearner

FEATURES = [500, 0.2, 0.1, 0.0, 0.9, 0]


def test_labels_come_from_evaluation_not_rules(tmp_path, monkeypatch):
    # Pretend one strategy always scores best: every label must follow the evaluation.
    winners = {"imputation": "median", "scaling": "minmax", "encoding": "label"}

    def fake_evaluate(task, strategy, X, y, task_type, scale_cols=None):
        return (1.0 if strategy == winners[task] else 0.0), None
    monkeypatch.setattr(train_meta_model, "evaluate_strategy", fake_evaluate)

    data = train_meta_model.create_training_data(n_datasets=4, seed=1, models_dir=str(tmp_path))

    for task, (X, y) in data.items():
        assert set(y) == {winners[task]}
        assert all(len(features) == len(FEATURE_NAMES) for features in X)


def test_real_evaluation_produces_known_strategies(tmp_path):
    data = train_meta_model.create_training_data(n_datasets=2, seed=2, models_dir=str(tmp_path))

    for task, (X, y) in data.items():
        assert y and set(y) <= set(STRATEGIES[task])


def test_summary_reports_top_k_hit_rates(tmp_path, capsys):
    rng = np.random.default_rng(0)
    X = rng.random((30, len(FEATURE_NAMES))).tolist()
    train_meta_model.summarize({"scaling": (X, ["standard", "robust", "minmax"] * 10)}, models_dir=str(tmp_path))

    out = capsys.readouterr().out
    assert "fixed rules top-1" in out and "meta-model top-1" in out and "top-2" in out


def test_stored_feedback_is_used_for_training(tmp_path, monkeypatch):
    monkeypatch.setattr(train_meta_model, "evaluate_strategy", lambda *a, **k: (0.5, None))
    learner = MetaLearner(str(tmp_path))
    learner.store_feedback("imputation", FEATURES, "knn")
    learner.store_feedback("imputation", FEATURES, "knn")

    X, y = train_meta_model.create_training_data(n_datasets=1, seed=3, models_dir=str(tmp_path))["imputation"]

    assert y.count("knn") >= 2 and FEATURES in X


def test_feedback_with_outdated_layout_is_archived(tmp_path):
    old_file = tmp_path / "scaling_feedback.csv"
    old_file.write_text("missing_ratio,skewness,best_strategy\n0.1,2.0,robust\n")
    learner = MetaLearner(str(tmp_path))

    learner.store_feedback("scaling", FEATURES, "standard")

    assert list(pd.read_csv(old_file).columns) == FEEDBACK_COLUMNS
    assert len(list((tmp_path / "legacy").glob("scaling_feedback_*.csv"))) == 1


def test_trained_models_are_saved_loaded_and_used(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((30, len(FEATURE_NAMES))).tolist()
    y = ["mean", "median", "knn"] * 10
    MetaLearner(str(tmp_path)).train({"imputation": (X, y)})

    learner = MetaLearner(str(tmp_path))
    learner.load_models()

    assert learner.is_fitted["imputation"]
    assert len(learner.predict_top_k("imputation", FEATURES, k=2)) == 2


def test_outdated_model_files_are_ignored(tmp_path):
    # A model fitted without feature names (the old layout) must not be trusted
    stale = RandomForestClassifier(n_estimators=5).fit(np.random.rand(10, 6), ["mean", "knn"] * 5)
    joblib.dump(stale, tmp_path / "imputation_model.pkl")

    learner = MetaLearner(str(tmp_path))
    learner.load_models()

    assert not learner.is_fitted["imputation"]

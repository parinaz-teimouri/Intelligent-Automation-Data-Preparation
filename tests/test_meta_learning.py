import pytest

from conftest import DATASETS
from data_preparation import AutoDataPreparer
from meta_learning import MetaLearner, STRATEGIES

# [n_samples, missing_ratio, skewness, kurtosis, unique_ratio, dtype_val]
NUMERIC_FEATURES = [500, 0.2, 0.1, 0.0, 0.9, 0]


@pytest.fixture
def learner(tmp_path):
    return MetaLearner(models_dir=str(tmp_path))  # no trained models -> fallback heuristic


@pytest.mark.parametrize("task", ["imputation", "scaling"])
def test_unfitted_learner_still_proposes_k_distinct_candidates(learner, task):
    candidates = learner.predict_top_k(task, NUMERIC_FEATURES, k=2)

    assert len(candidates) == 2 and len(set(candidates)) == 2
    assert set(candidates) <= set(STRATEGIES[task])
    # The heuristic's choice stays first.
    assert candidates[0] == learner._fallback_heuristic(task, NUMERIC_FEATURES)


def test_excluded_strategies_are_never_proposed(learner):
    ranked = learner.rank_strategies("imputation", NUMERIC_FEATURES, exclude={"knn"})

    assert "knn" not in ranked
    assert learner.predict("imputation", NUMERIC_FEATURES, exclude={"knn"}) != "knn"


def test_excluding_every_strategy_raises(learner):
    with pytest.raises(ValueError):
        learner.rank_strategies("encoding", NUMERIC_FEATURES, exclude=set(STRATEGIES["encoding"]))


def test_knn_is_pruned_for_large_datasets(run_pipeline, monkeypatch):
    monkeypatch.setattr(AutoDataPreparer, "KNN_MAX_ROWS", 50)

    result = run_pipeline(DATASETS["classification"][0](), "segment")

    assert "knn" not in result.winner("Imputation")["details"]["candidates"]
    assert any(d["decision"] == "KNN Excluded" for d in result.decisions("Imputation"))


def test_default_models_dir_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    learner = MetaLearner()

    assert not learner.models_dir.startswith(str(tmp_path))
    assert learner.models_dir.endswith("meta_models")

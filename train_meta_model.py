"""
Generates meta-training data and trains the meta-models.

Labels are empirical: for every synthetic dataset, each candidate strategy is
cross-validated with the same downstream evaluation the pipeline uses
(data_preparation.evaluate_strategy), and the best-scoring strategy becomes the
label. Feedback stored by real pipeline runs (meta_models/<task>_feedback.csv)
is added to the training data, closing the learning loop.

Usage: python train_meta_model.py [--datasets 40] [--seed 0] [--models-dir meta_models]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

from data_preparation import evaluate_strategy
from meta_learning import FEATURE_NAMES, STRATEGIES, MetaFeatureExtractor, MetaLearner


def generate_base_dataset(rng):
    """A random supervised dataset: varied size, width, task type and noise."""
    n_samples = int(rng.integers(150, 1500))
    n_features = int(rng.integers(4, 12))
    n_informative = max(2, int(n_features * 0.6))
    seed = int(rng.integers(1_000_000))

    if rng.random() < 0.5:
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                   n_redundant=0, n_classes=int(rng.integers(2, 4)), n_clusters_per_class=1,
                                   flip_y=rng.uniform(0, 0.1), random_state=seed)
        task_type = "classification"
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               noise=rng.uniform(1, 30), random_state=seed)
        task_type = "regression"

    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(n_features)])
    return df, pd.Series(y), task_type


def skew_some_columns(df, rng):
    """Makes a random subset of columns skewed or heavy-tailed, as real data often is."""
    df = df.copy()
    for col in rng.choice(df.columns, size=int(rng.integers(0, len(df.columns) + 1)), replace=False):
        kind = rng.choice(["exp", "square", "outliers"])
        if kind == "exp":
            df[col] = np.exp(df[col])
        elif kind == "square":
            df[col] = df[col] ** 2
        else:
            idx = rng.choice(len(df), size=max(1, len(df) // 50), replace=False)
            df.loc[df.index[idx], col] *= rng.uniform(10, 100)
    return df


def best_strategy(task, X, y, task_type, scale_cols=None):
    """Label = the candidate with the best cross-validated downstream score (None if all fail)."""
    scores = {}
    for strategy in STRATEGIES[task]:
        score, _ = evaluate_strategy(task, strategy, X, y, task_type, scale_cols)
        if score is not None:
            scores[strategy] = score
    return max(scores, key=scores.get) if scores else None


def imputation_example(rng):
    df, y, task_type = generate_base_dataset(rng)
    df = skew_some_columns(df, rng)
    missing_rate = rng.uniform(0.05, 0.5)
    for col in rng.choice(df.columns, size=int(rng.integers(1, len(df.columns) + 1)), replace=False):
        df.loc[rng.random(len(df)) < missing_rate, col] = np.nan
    # Same column the pipeline describes: the one with the most missing values
    worst_col = df.isnull().sum().idxmax()
    return MetaFeatureExtractor.extract_features(df, worst_col), best_strategy("imputation", df, y, task_type)


def scaling_example(rng):
    df, y, task_type = generate_base_dataset(rng)
    df = skew_some_columns(df, rng)
    # Same column the pipeline describes: the one with the largest absolute skew
    max_skew_col = df.skew().abs().idxmax()
    label = best_strategy("scaling", df, y, task_type, scale_cols=list(df.columns))
    return MetaFeatureExtractor.extract_features(df, max_skew_col), label


def encoding_example(rng):
    df, y, task_type = generate_base_dataset(rng)
    col = rng.choice(df.columns)
    n_categories = int(rng.integers(2, 60))
    codes = pd.cut(df[col], bins=n_categories, labels=False)
    # Category names are sometimes unrelated to the underlying order, as with most real labels
    names = rng.permutation(n_categories) if rng.random() < 0.5 else np.arange(n_categories)
    df[col] = [f"cat_{names[c]:02d}" for c in codes]
    return MetaFeatureExtractor.extract_features(df, col), best_strategy("encoding", df, y, task_type)


def create_training_data(n_datasets=40, seed=0, models_dir=None):
    """Generates n_datasets labelled examples per task and appends stored feedback."""
    rng = np.random.default_rng(seed)
    generators = {"imputation": imputation_example, "scaling": scaling_example, "encoding": encoding_example}
    learner = MetaLearner(models_dir)
    training_data = {}

    for task, make_example in generators.items():
        print(f"Generating {n_datasets} {task} datasets...")
        X, y = [], []
        for _ in range(n_datasets):
            features, label = make_example(rng)
            if label is not None:
                X.append(features)
                y.append(label)

        X_feedback, y_feedback = learner.load_feedback(task)
        if X_feedback:
            print(f"  + {len(X_feedback)} feedback examples from pipeline runs")
        training_data[task] = (X + X_feedback, y + y_feedback)

    return training_data


def _top_k_hit_rate(rankings, labels, k):
    return float(np.mean([label in ranking[:k] for ranking, label in zip(rankings, labels)]))


def summarize(training_data, models_dir=None):
    """
    How well each approach finds the best strategy, cross-validated:
    top-1 = best strategy ranked first; top-2 = best strategy among the 2 candidates the
    pipeline evaluates. Compared with always guessing the most common label and the fixed rules.
    """
    rules = MetaLearner(models_dir)  # no models loaded: ranks with the fixed heuristic

    for task, (X, y) in training_data.items():
        counts = pd.Series(y).value_counts()
        print(f"{task}: {len(y)} examples, labels {counts.to_dict()}")
        if len(y) < 10 or counts.min() < 2 or len(counts) < 2:
            print("  (too few examples for a cross-validated check)")
            continue

        proba = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=42),
                                  pd.DataFrame(X, columns=FEATURE_NAMES), y,
                                  cv=min(5, int(counts.min())), method="predict_proba")
        classes = np.array(sorted(counts.index))
        model_rankings = [list(classes[np.argsort(p)[::-1]]) for p in proba]
        rule_rankings = [rules.rank_strategies(task, x) for x in X]

        print(f"  majority guess top-1 {counts.max() / len(y):.2f}"
              f" | fixed rules top-1 {_top_k_hit_rate(rule_rankings, y, 1):.2f}, top-2 {_top_k_hit_rate(rule_rankings, y, 2):.2f}"
              f" | meta-model top-1 {_top_k_hit_rate(model_rankings, y, 1):.2f}, top-2 {_top_k_hit_rate(model_rankings, y, 2):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=int, default=40, help="synthetic datasets per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", default=None, help="where models and feedback live (default: meta_models/)")
    args = parser.parse_args()

    training_data = create_training_data(n_datasets=args.datasets, seed=args.seed, models_dir=args.models_dir)
    summarize(training_data, models_dir=args.models_dir)

    learner = MetaLearner(args.models_dir)
    learner.train(training_data)
    print("Meta-models trained successfully.")

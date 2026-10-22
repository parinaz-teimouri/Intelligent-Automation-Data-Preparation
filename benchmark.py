"""
Benchmark: static preprocessing vs. the adaptive data preparation pipeline.

Configurations (all share deduplication, outlier handling and date expansion):
  static             mean imputation, one-hot encoding, standard scaling; no value cleaning
  static + cleaning  the same fixed strategies, plus value standardisation and identifier removal
  meta-model only    cleaning + strategies predicted by the meta-model (or fixed rules), no evaluation
  full system        cleaning + Top-K candidates scored against the target (the proposed method)

Protocol, repeated with different random splits:
  1. Split each dataset into a selection half and an evaluation half.
  2. The full system chooses its strategies on the selection half only, then those
     choices are applied to the evaluation half. The other configurations never see
     the selection half, so no configuration is scored on rows it was tuned on.
  3. Each prepared evaluation half is scored with two downstream models (a linear model and a
     random forest) using 5-fold cross-validation: macro F1 for classification, R2 for regression.

Usage: python benchmark.py [--repeats 3] [--no-openml] [--output-dir benchmark_results]
"""
import argparse
import contextlib
import io
import os
import tempfile
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from data_preparation import AutoDataPreparer
from meta_learning import MetaLearner

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC = {"imputation": "mean", "scaling": "standard", "encoding": "onehot"}
CONFIGS = ["static", "static + cleaning", "meta-model only", "full system"]


# --- Datasets ----------------------------------------------------------------------

def inject_defects(df, target, seed):
    """
    Adds the kinds of defects found in real exports to a clean dataset: skewed columns,
    missing values (random and value-dependent), numbers stored as currency text, a category
    written in several spellings, a yes/no column and an ID column. Informative columns are
    converted, so cleaning them up matters for the downstream model.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    features = [c for c in df.columns if c != target]
    cols = list(rng.permutation(features))

    # Skew a third of the columns (standardised first so exp() stays finite)
    for col in cols[: len(cols) // 3]:
        z = (df[col] - df[col].mean()) / df[col].std()
        df[col] = np.exp(z)

    # Missing values: random in some columns, and value-dependent in one column
    for col in cols[: len(cols) // 2]:
        df.loc[rng.random(len(df)) < 0.2, col] = np.nan
    mar_source, mar_col = cols[-1], cols[-2]
    df.loc[(df[mar_source] > df[mar_source].quantile(0.7)) & (rng.random(len(df)) < 0.6), mar_col] = np.nan

    # Numbers stored as currency text
    money_col = cols[-3]
    formats = rng.choice(["${:.2f}", "USD {:.2f}", "{:.2f} EUR", "{:.2f}"], size=len(df))
    df[money_col] = [f.format(v) if pd.notna(v) else np.nan for f, v in zip(formats, df[money_col])]

    # A category in several spellings, replacing an informative numeric column
    level_col = cols[-4]
    levels = pd.qcut(df[level_col], 3, labels=["low", "medium", "high"]).astype(str)
    spellings = {"low": ["low", "Low", "LOW", "l"], "medium": ["medium", "Medium", "MEDIUM", "m"],
                 "high": ["high", "High", "HIGH", "h"]}
    df[f"{level_col}_level"] = [rng.choice(spellings[v]) if v in spellings else np.nan for v in levels]

    # A yes/no flag, replacing another informative column
    flag_col = cols[-5]
    above = df[flag_col] > df[flag_col].median()
    df[f"{flag_col}_flag"] = [rng.choice(["Y", "yes", "True", "1"]) if a else rng.choice(["N", "no", "False", "0"])
                              for a in above]
    df = df.drop(columns=[level_col, flag_col])

    df.insert(0, "record_id", [f"ID-{i:05d}" for i in range(len(df))])
    return df


def _sklearn_dataset(loader, target, task_type, name):
    data = loader(as_frame=True)
    df = data.frame.rename(columns={data.target.name: target})
    if task_type == "classification":
        df[target] = df[target].map(lambda i: data.target_names[i])
    return name, inject_defects(df, target, seed=0), target, task_type, \
        f"scikit-learn {name.split(' ')[0]} with injected defects"


def _openml_dataset(name, version, target, task_type, drop=(), sample=None):
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # python.org builds on macOS lack CA certificates
    from sklearn.datasets import fetch_openml
    frame = fetch_openml(name, version=version, as_frame=True).frame.drop(columns=list(drop))
    if sample and len(frame) > sample:
        frame = frame.sample(sample, random_state=0)
    return f"{name} (OpenML)", frame.reset_index(drop=True), target, task_type, f"OpenML '{name}' v{version}, real data"


def load_datasets(use_openml=True):
    datasets = [
        _sklearn_dataset(load_breast_cancer, "diagnosis", "classification", "breast_cancer (defects)"),
        _sklearn_dataset(load_wine, "cultivar", "classification", "wine (defects)"),
        _sklearn_dataset(load_diabetes, "progression", "regression", "diabetes (defects)"),
    ]
    freelancers = pd.read_csv(os.path.join(REPO_DIR, "input", "global_freelancers_raw.csv"))
    datasets.append(("freelancers (repo)", freelancers, "rating", "regression",
                     "input/global_freelancers_raw.csv, messy real-world style export"))

    if use_openml:
        openml = [
            # 'boat' and 'body' are recorded after the fact (lifeboat / body number) and give away the target
            ("titanic", 1, "survived", "classification", ("boat", "body"), None),
            ("adult", 2, "class", "classification", (), 3000),
            ("credit-g", 1, "class", "classification", (), None),
        ]
        for name, version, target, task_type, drop, sample in openml:
            try:
                datasets.append(_openml_dataset(name, version, target, task_type, drop, sample))
            except Exception as e:
                print(f"Skipping OpenML dataset '{name}': {type(e).__name__}: {e}")

    # Rows without a target cannot be scored by any configuration
    return [(name, df[df[target].notna()].reset_index(drop=True), target, task_type, source)
            for name, df, target, task_type, source in datasets]


# --- Running configurations ----------------------------------------------------------

def prepare(df, target, workdir, label, models_dir, **options):
    """Runs the pipeline on a DataFrame; returns (ML-ready DataFrame, preparer)."""
    input_path = os.path.join(workdir, f"{label}.csv")
    df.to_csv(input_path, index=False)
    preparer = AutoDataPreparer(input_path, target_col=target, models_dir=models_dir,
                                record_feedback=False, **options)
    with contextlib.redirect_stdout(io.StringIO()):
        preparer.run_pipeline(os.path.join(workdir, f"processed_{label}.csv"))
    return pd.read_csv(os.path.join(workdir, f"processed_{label}.csv")), preparer


def run_config(config, selection, evaluation, target, workdir, models_dir):
    if config == "static":
        return prepare(evaluation, target, workdir, "static", models_dir,
                       fixed_strategies=STATIC, standardize=False, optimize=False)[0], {}
    if config == "static + cleaning":
        return prepare(evaluation, target, workdir, "cleaning", models_dir,
                       fixed_strategies=STATIC, optimize=False)[0], {}
    if config == "meta-model only":
        processed, preparer = prepare(evaluation, target, workdir, "meta", models_dir, optimize=False)
        return processed, preparer.chosen_strategies
    if config == "full system":
        _, selector = prepare(selection, target, workdir, "select", models_dir, optimize=True)
        chosen = {task: selector.chosen_strategies[task] for task in ("imputation", "scaling")
                  if task in selector.chosen_strategies}
        processed, preparer = prepare(evaluation, target, workdir, "full", models_dir,
                                      fixed_strategies=chosen, optimize=False)
        return processed, {**preparer.chosen_strategies, **chosen}
    raise ValueError(config)


def downstream_scores(processed, target, task_type, seed):
    """Cross-validated score of two downstream models on a prepared dataset."""
    X, y = processed.drop(columns=[target]), processed[target]
    if task_type == "classification":
        models = {"linear": LogisticRegression(max_iter=5000),
                  "random forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)}
        cv, scoring = StratifiedKFold(5, shuffle=True, random_state=seed), "f1_macro"
    else:
        models = {"linear": Ridge(),
                  "random forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)}
        cv, scoring = KFold(5, shuffle=True, random_state=seed), "r2"
    return {name: float(cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())
            for name, model in models.items()}


def run_benchmark(datasets, repeats, models_dir=None):
    rows = []
    for name, df, target, task_type, _ in datasets:
        for seed in range(repeats):
            stratify = df[target] if task_type == "classification" else None
            selection, evaluation = train_test_split(df, test_size=0.5, random_state=seed, stratify=stratify)
            with tempfile.TemporaryDirectory() as workdir:
                for config in CONFIGS:
                    start = time.time()
                    processed, chosen = run_config(config, selection, evaluation, target, workdir, models_dir)
                    prep_seconds = time.time() - start
                    scores = downstream_scores(processed, target, task_type, seed)
                    for model, score in scores.items():
                        rows.append({"dataset": name, "task": task_type, "repeat": seed, "config": config,
                                     "model": model, "score": score, "prep_seconds": prep_seconds,
                                     "features": processed.shape[1] - 1,
                                     "imputation": chosen.get("imputation", STATIC["imputation"] if "static" in config else ""),
                                     "scaling": chosen.get("scaling", STATIC["scaling"] if "static" in config else "")})
            print(f"  {name}: repeat {seed + 1}/{repeats} done")
    return pd.DataFrame(rows)


# --- Reporting -------------------------------------------------------------------------

def summary_tables(results):
    """Per dataset and configuration: mean (std) over repeats of the score averaged across both models."""
    per_repeat = results.groupby(["dataset", "task", "config", "repeat"], sort=False)["score"].mean().reset_index()
    agg = per_repeat.groupby(["dataset", "task", "config"], sort=False)["score"].agg(["mean", "std"]).reset_index()

    table = agg.pivot(index=["dataset", "task"], columns="config", values="mean")[CONFIGS]
    stds = agg.pivot(index=["dataset", "task"], columns="config", values="std")[CONFIGS].fillna(0)
    ranks = table.rank(axis=1, ascending=False).mean()

    by_model = results.groupby(["dataset", "model", "config"], sort=False)["score"].mean().unstack("config")[CONFIGS]
    timing = results.groupby("config")["prep_seconds"].mean()[CONFIGS]
    features = results.groupby(["dataset", "config"], sort=False)["features"].mean().unstack("config")[CONFIGS]
    return table, stds, ranks, by_model, timing, features, head_to_head(per_repeat)


def head_to_head(per_repeat, tolerance=1e-9):
    """Paired comparison on identical splits: how often configuration A beats configuration B."""
    paired = per_repeat.pivot_table(index=["dataset", "repeat"], columns="config", values="score")
    rows = []
    for a, b in [("static + cleaning", "static"), ("full system", "static + cleaning"),
                 ("full system", "meta-model only"), ("full system", "static")]:
        diff = paired[a] - paired[b]
        rows.append({"comparison": f"{a} vs {b}", "wins": int((diff > tolerance).sum()),
                     "ties": int((diff.abs() <= tolerance).sum()), "losses": int((diff < -tolerance).sum()),
                     "mean_difference": float(diff.mean())})
    return pd.DataFrame(rows)


def to_markdown(table, stds, ranks, by_model, timing, features, pairs, datasets, repeats, meta_status):
    def fmt_row(cells):
        return "| " + " | ".join(cells) + " |"

    lines = [
        "# Benchmark results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `benchmark.py` "
        f"({repeats} repeats, scikit-learn {sklearn.__version__}, pandas {pd.__version__}). "
        f"Meta-models: {meta_status}.",
        "",
        "Score = macro F1 (classification) or R² (regression) on the evaluation half, 5-fold CV, "
        "averaged over a linear model and a random forest. Mean (std over repeats); best per row in bold.",
        "",
        fmt_row(["Dataset", "Task"] + CONFIGS),
        fmt_row(["---", "---"] + ["---:"] * len(CONFIGS)),
    ]
    for (dataset, task), row in table.iterrows():
        best = row.max()
        cells = []
        for config in CONFIGS:
            cell = f"{row[config]:.3f} ({stds.loc[(dataset, task), config]:.3f})"
            cells.append(f"**{cell}**" if np.isclose(row[config], best) else cell)
        lines.append(fmt_row([dataset, task] + cells))
    lines.append(fmt_row(["**Mean rank** (1 = best)", ""] + [f"{ranks[c]:.2f}" for c in CONFIGS]))
    lines.append(fmt_row(["Mean preparation time (s)", ""] + [f"{timing[c]:.2f}" for c in CONFIGS]))

    lines += ["", "## Head to head", "",
              "Each dataset × repeat is one paired comparison on the identical split (score averaged over both models).",
              "", fmt_row(["Comparison", "Wins", "Ties", "Losses", "Mean difference"]),
              fmt_row(["---", "---:", "---:", "---:", "---:"])]
    for _, pair in pairs.iterrows():
        lines.append(fmt_row([pair["comparison"], str(pair["wins"]), str(pair["ties"]), str(pair["losses"]),
                              f"{pair['mean_difference']:+.4f}"]))

    lines += ["", "## Per downstream model", "", fmt_row(["Dataset", "Model"] + CONFIGS),
              fmt_row(["---", "---"] + ["---:"] * len(CONFIGS))]
    for (dataset, model), row in by_model.iterrows():
        lines.append(fmt_row([dataset, model] + [f"{row[c]:.3f}" for c in CONFIGS]))

    lines += ["", "## Number of features after preparation", "", fmt_row(["Dataset"] + CONFIGS),
              fmt_row(["---"] + ["---:"] * len(CONFIGS))]
    for dataset, row in features.iterrows():
        lines.append(fmt_row([dataset] + [f"{row[c]:.0f}" for c in CONFIGS]))

    lines += ["", "## Datasets", ""]
    lines += [f"- **{name}**: {len(df)} rows, target `{target}` ({task_type}); {source}"
              for name, df, target, task_type, source in datasets]
    lines += [
        "",
        "## Notes",
        "",
        "- The full system chooses imputation and scaling on the selection half and is scored on the "
        "disjoint evaluation half, so its choices are not tuned on the rows it is scored on.",
        "- Within the evaluation half, preprocessing is fitted on all its rows before cross-validation. "
        "This is identical for every configuration, so comparisons are fair, but absolute scores are slightly optimistic.",
        "- Defects in the scikit-learn datasets are injected synthetically (`inject_defects`); "
        "the OpenML and repository datasets are used as they are.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3, help="random selection/evaluation splits per dataset")
    parser.add_argument("--no-openml", action="store_true", help="skip datasets downloaded from OpenML")
    parser.add_argument("--models-dir", default=None, help="meta-models to use (default: meta_models/)")
    parser.add_argument("--output-dir", default=os.path.join(REPO_DIR, "benchmark_results"))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")  # convergence / ill-defined-metric noise from small CV folds
    learner = MetaLearner(args.models_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        learner.load_models()
    meta_status = ("trained models" if all(learner.is_fitted.values())
                   else "none trained, fixed rules used (run train_meta_model.py)")

    datasets = load_datasets(use_openml=not args.no_openml)
    print(f"Benchmarking {len(datasets)} datasets x {len(CONFIGS)} configurations x {args.repeats} repeats "
          f"(meta-models: {meta_status})")
    results = run_benchmark(datasets, args.repeats, args.models_dir)

    tables = summary_tables(results)
    markdown = to_markdown(*tables, datasets, args.repeats, meta_status)
    os.makedirs(args.output_dir, exist_ok=True)
    results.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    with open(os.path.join(args.output_dir, "RESULTS.md"), "w") as f:
        f.write(markdown)
    print("\n" + markdown)
    print(f"Saved {os.path.join(args.output_dir, 'results.csv')} and RESULTS.md")

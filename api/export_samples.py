"""
Pre-computes the demo's sample results as static files, so the sample datasets work instantly
without the API (e.g. while a free backend instance is waking up).

  <out>/samples/index.json                     the catalog, with the target options per sample
  <out>/samples/<id>/<target|none>/result.json  the result, downloads replaced by file paths
  <out>/samples/<id>/<target|none>/*.csv|json   clean.csv, ml.csv, report.json, shopify.csv
  <out>/benchmark.json                          headline numbers from benchmark_results/results.csv

Usage: python -m api.export_samples [--out web/public/data]
"""
import argparse
import json
import os
import shutil

import pandas as pd

from api.service import prepare_file

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_FILES = {"clean_csv": "clean.csv", "ml_csv": "ml.csv", "report_json": "report.json", "shopify_csv": "shopify.csv"}
VENDORS = {"fashion": "Atelier Kova"}


def export_samples(out_dir, url_prefix="/data"):
    with open(os.path.join(REPO_DIR, "samples", "catalog.json")) as f:
        catalog = json.load(f)
    samples_dir = os.path.join(out_dir, "samples")
    shutil.rmtree(samples_dir, ignore_errors=True)

    index = []
    for entry in catalog:
        variants = []
        for target in [None] + entry["targets"]:
            key = target or "none"
            result = prepare_file(os.path.join(REPO_DIR, entry["file"]), entry["name"], target,
                                  vendor=VENDORS.get(entry["id"]))
            folder = os.path.join(samples_dir, entry["id"], key)
            os.makedirs(folder, exist_ok=True)
            for field, filename in DOWNLOAD_FILES.items():
                content = result["downloads"][field]
                if content is None:
                    result["downloads"][field] = None
                    continue
                with open(os.path.join(folder, filename), "w") as f:
                    f.write(content)
                result["downloads"][field] = f"{url_prefix}/samples/{entry['id']}/{key}/{filename}"
            with open(os.path.join(folder, "result.json"), "w") as f:
                json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
            variants.append(key)
            print(f"  {entry['id']}/{key}: {result['stats']['standardised']} values standardised, "
                  f"{result['stats']['filled']} gaps filled")
        index.append({**entry, "variants": variants})

    with open(os.path.join(samples_dir, "index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)


def export_benchmark(out_dir):
    """Paired comparisons on identical splits, as reported in benchmark_results/RESULTS.md."""
    results = pd.read_csv(os.path.join(REPO_DIR, "benchmark_results", "results.csv"))
    per_repeat = results.groupby(["dataset", "repeat", "config"])["score"].mean().unstack("config")

    def pair(a, b):
        diff = per_repeat[a] - per_repeat[b]
        return {"wins": int((diff > 1e-9).sum()), "ties": int((diff.abs() <= 1e-9).sum()),
                "losses": int((diff < -1e-9).sum()), "mean_difference": float(diff.mean())}

    benchmark = {
        "datasets": sorted(results["dataset"].unique()),
        "repeats": int(results["repeat"].nunique()),
        "cleaning_vs_static": pair("static + cleaning", "static"),
        "selection_vs_cleaning": pair("full system", "static + cleaning"),
        "full_vs_static": pair("full system", "static"),
    }
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(benchmark, f, indent=1)
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_DIR, "web", "public", "data"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("Exporting sample results ...")
    export_samples(args.out)
    print("Benchmark:", export_benchmark(args.out))

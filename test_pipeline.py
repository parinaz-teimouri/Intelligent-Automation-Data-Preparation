"""
Runs the full preparation pipeline on one data file and previews the result.

Usage: python test_pipeline.py <data.csv|data.xlsx> [--target COLUMN] [--output-dir DIR]

With --target the pipeline runs in optimisation mode (candidate strategies are
scored against the target); without it, strategies are only predicted.
"""
import argparse
import os
import sys

import pandas as pd

from data_preparation import AutoDataPreparer


def test_pipeline(file_path, target_col=None, output_dir="."):
    print(f"Starting pipeline for: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False

    preparer = AutoDataPreparer(file_path, target_col=target_col)

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{base_name}.csv")

    # Run the full pipeline
    try:
        preparer.run_pipeline(output_path)

        # Load and check the processed data
        if os.path.exists(output_path):
            processed_data = pd.read_csv(output_path)
            print("\nProcessed Data Preview (ML Ready):")
            print(processed_data.head())
        print(f"\nPipeline completed successfully.")
        return True
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="CSV or Excel file to prepare")
    parser.add_argument("--target", default=None, help="target column for supervised evaluation (optional)")
    parser.add_argument("--output-dir", default=".", help="where processed data and the report are written")
    args = parser.parse_args()

    sys.exit(0 if test_pipeline(args.file, args.target, args.output_dir) else 1)

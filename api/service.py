"""Runs the pipeline on one file in a throw-away directory and returns the app's result JSON."""
import contextlib
import io
import os
import tempfile

from api.results import build_result
from data_preparation import AutoDataPreparer


def prepare_file(path, name, target=None, vendor=None, models_dir=None):
    """
    Prepares the file at `path`. Nothing is kept: outputs live in a temporary directory that is
    deleted before returning, and evaluation feedback is not recorded.
    """
    with tempfile.TemporaryDirectory() as workdir:
        output = os.path.join(workdir, "processed_data.csv")
        preparer = AutoDataPreparer(path, target_col=target, models_dir=models_dir, record_feedback=False)
        with contextlib.redirect_stdout(io.StringIO()):
            preparer.run_pipeline(output)
        with open(output) as f:
            ml_csv = f.read()
        with open(os.path.join(workdir, "report_processed_data.json")) as f:
            report_json = f.read()
    return build_result(preparer, name, ml_csv, report_json, vendor=vendor)

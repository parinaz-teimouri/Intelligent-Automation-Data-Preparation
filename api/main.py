"""
HTTP API for the demo app.

  GET  /api/health    liveness check
  GET  /api/samples   the sample datasets (samples/catalog.json)
  POST /api/prepare   multipart upload: file (CSV or Excel), optional target -> result JSON

Uploads are processed in a temporary directory and deleted afterwards; nothing is stored.
Run locally with: uvicorn api.main:app --reload
"""
import asyncio
import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.service import prepare_file

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
MAX_ROWS = int(os.environ.get("MAX_ROWS", 20_000))
TIMEOUT_SECONDS = float(os.environ.get("PREPARE_TIMEOUT_SECONDS", 60))
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(title="Data Preparation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# A small instance should not run several heavy jobs at once
jobs = asyncio.Semaphore(int(os.environ.get("MAX_PARALLEL_JOBS", 2)))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/samples")
def samples():
    with open(os.path.join(REPO_DIR, "samples", "catalog.json")) as f:
        return json.load(f)


@app.post("/api/prepare")
async def prepare(file: UploadFile = File(...), target: str | None = Form(None)):
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Please upload a CSV or Excel file (.csv, .xlsx).")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not content.strip():
        raise HTTPException(400, "The file is empty.")
    if extension == ".csv" and content.count(b"\n") > MAX_ROWS + 1:
        raise HTTPException(413, f"The file has more than {MAX_ROWS:,} rows.")

    name = os.path.splitext(os.path.basename(file.filename))[0] or "upload"
    async with jobs:
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, f"upload{extension}")
            with open(path, "wb") as f:
                f.write(content)
            try:
                return await asyncio.wait_for(
                    run_in_threadpool(prepare_file, path, name, target or None), TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                raise HTTPException(504, "Preparing this file took too long. Try a smaller file.")
            except ValueError as e:
                # Unknown target column, unreadable file, ...
                raise HTTPException(400, str(e))
            except Exception:
                raise HTTPException(422, "This file could not be prepared. Check that it is a table with a header row.")

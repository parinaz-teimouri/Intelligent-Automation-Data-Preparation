# API for the demo app: docker build -t data-prep-api . && docker run -p 8000:8000 data-prep-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

COPY data_preparation.py meta_learning.py ./
COPY api/ api/
COPY samples/catalog.json samples/catalog.json
COPY meta_models/ meta_models/

# Run without root; the pipeline only writes to temporary directories
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT', '8000') + '/api/health')"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

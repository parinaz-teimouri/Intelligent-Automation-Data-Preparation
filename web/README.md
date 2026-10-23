# Demo web app

A static Next.js site for the data preparation pipeline: sample datasets run from pre-computed
results in `public/data/`, uploads are sent to the FastAPI backend in `../api`.

## Develop

```bash
pnpm install
pnpm dev                         # http://localhost:3000
```

Start the API for uploads from the repository root: `uvicorn api.main:app --reload` (port 8000).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend that handles uploads |
| `NEXT_PUBLIC_BASE_PATH` | (empty) | Sub-path when served from one, e.g. `/repository-name` on GitHub Pages |
| `NEXT_PUBLIC_BENCHMARK_URL` | the repository's `benchmark_results/RESULTS.md` | Target of the "Read the full benchmark" link |

## Sample data

`public/data/` is generated from the pipeline; regenerate it after changing the pipeline or the samples:

```bash
python -m api.export_samples     # from the repository root
```

## Build

```bash
pnpm build                       # static site in out/
```

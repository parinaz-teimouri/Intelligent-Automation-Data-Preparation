import type { Benchmark, DownloadKey, Result, SampleEntry } from "./types";

/** Set when the site is served from a sub-path (e.g. GitHub Pages: /repository-name). */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, readonly unavailable = false) {
    super(message);
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_PATH}${path}`);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json() as Promise<T>;
}

export const loadSamples = () => getJson<SampleEntry[]>("/data/samples/index.json");
export const loadBenchmark = () => getJson<Benchmark>("/data/benchmark.json");
export const loadSampleResult = (id: string, variant: string) =>
  getJson<Result>(`/data/samples/${id}/${variant}/result.json`);

export async function prepareUpload(file: File, target: string | null): Promise<Result> {
  const body = new FormData();
  body.append("file", file);
  if (target) body.append("target", target);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/prepare`, { method: "POST", body });
  } catch {
    throw new ApiError("unavailable", true);
  }
  if (!response.ok) {
    const detail = await response.json().then((d) => d.detail as string).catch(() => "");
    throw new ApiError(detail || `HTTP ${response.status}`, response.status >= 502);
  }
  return response.json() as Promise<Result>;
}

/** Text of a download: fetched for sample results, inline for uploads. */
export async function downloadText(result: Result, key: DownloadKey, fromSample: boolean): Promise<string | null> {
  const value = result.downloads[key];
  if (value === null) return null;
  if (!fromSample) return value;
  const response = await fetch(`${BASE_PATH}${value}`);
  return response.ok ? response.text() : null;
}

export async function saveDownload(result: Result, key: DownloadKey, fromSample: boolean, filename: string) {
  const text = await downloadText(result, key, fromSample);
  if (text === null) return;
  const type = key === "report_json" ? "application/json" : "text/csv";
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

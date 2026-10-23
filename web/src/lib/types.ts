export type Lang = "en" | "de";
export type L10n = Record<Lang, string>;

export interface Issue {
  id: string;
  count: number;
  columns: string[];
  label: L10n;
}

export interface Step {
  id: string;
  seconds: number;
  label: L10n;
}

export type CellKind = "ok" | "fixed" | "removed" | "note";

export interface Cell {
  before: string | null;
  after: string | null;
  kind: CellKind;
  reason: string | null;
}

export interface PreviewRow {
  row: number;
  removed: boolean;
  cells: Cell[];
}

export interface Decision {
  tag: L10n;
  title: L10n;
  body: L10n;
}

export type DownloadKey = "clean_csv" | "ml_csv" | "report_json" | "shopify_csv";
export type StatKey = "quality" | "standardised" | "duplicates" | "filled";

export interface Result {
  dataset: {
    name: L10n;
    rows: number;
    columns: number;
    rows_out: number;
    columns_ml: number;
    column_names: string[];
    target_options: string[];
  };
  target: string | null;
  task_type: "classification" | "regression" | null;
  steps: Step[];
  issues: Issue[];
  stats: {
    quality_before: number;
    quality_after: number;
    standardised: number;
    filled: number;
    duplicates: number;
  };
  stat_labels: Record<StatKey, L10n>;
  preview: {
    columns: string[];
    rows: PreviewRow[];
    reasons: Record<string, L10n>;
  };
  decisions: Decision[];
  summary: L10n & { kind: "automatic" | "ai" };
  strategies: { imputation: string | null; scaling: string | null };
  /** Sample results hold file paths; upload results hold the file contents. */
  downloads: Record<DownloadKey, string | null>;
}

export interface SampleEntry {
  id: string;
  file: string;
  shopify: boolean;
  targets: string[];
  name: L10n;
  domain: L10n;
  source: L10n;
  licence: string;
  url: string | null;
  rows: number;
  columns: number;
  variants: string[];
}

export interface PairedComparison {
  wins: number;
  ties: number;
  losses: number;
  mean_difference: number;
}

export interface Benchmark {
  datasets: string[];
  repeats: number;
  cleaning_vs_static: PairedComparison;
  selection_vs_cleaning: PairedComparison;
  full_vs_static: PairedComparison;
}

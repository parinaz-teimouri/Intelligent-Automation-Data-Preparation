"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, loadSampleResult, loadSamples, prepareUpload } from "@/lib/data";
import { formatNumber, formatPercent, formatSeconds } from "@/lib/i18n";
import { useLang } from "@/lib/lang";
import type { Result, SampleEntry, StatKey } from "@/lib/types";
import { Decisions, Export, Summary } from "./Panels";
import { Preview, type CellPosition, type View } from "./Preview";
import styles from "./Workspace.module.css";

type Tab = "data" | "why" | "summary" | "export";
type Problem = { kind: "unavailable" } | { kind: "message"; text: string } | { kind: "generic" };

const UPLOAD = "upload";
const MAX_TARGET_CHIPS = 6;

/** Visible pacing for the live pipeline: real step times are often a few milliseconds. */
function stepDelay(seconds: number) {
  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  return reduced ? 60 : Math.min(700, Math.max(280, seconds * 1000));
}

function Tick() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" className={styles.tick} aria-hidden="true">
      <path d="M3 8.5l3 3 7-7" />
    </svg>
  );
}

export function Workspace() {
  const { lang, t } = useLang();
  const [samples, setSamples] = useState<SampleEntry[] | null>(null);
  const [datasetId, setDatasetId] = useState("fashion");
  const [variant, setVariant] = useState("none");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(true);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);

  const [stage, setStage] = useState(0);
  const [running, setRunning] = useState(false);
  const [ran, setRan] = useState(false);
  const [view, setView] = useState<View>("before");
  const [tab, setTab] = useState<Tab>("data");
  const [selected, setSelected] = useState<CellPosition | null>(null);

  const request = useRef(0);
  const timers = useRef<number[]>([]);

  const resetRun = useCallback(() => {
    timers.current.forEach(window.clearTimeout);
    timers.current = [];
    setStage(0);
    setRunning(false);
    setRan(false);
    setView("before");
    setTab("data");
    setSelected(null);
  }, []);

  const load = useCallback(
    (task: () => Promise<Result>) => {
      const token = ++request.current;
      resetRun();
      setLoading(true);
      setProblem(null);
      task()
        .then((next) => token === request.current && setResult(next))
        .catch((error) => {
          if (token !== request.current) return;
          setResult(null);
          if (error instanceof ApiError) {
            setProblem(error.unavailable ? { kind: "unavailable" } : { kind: "message", text: error.message });
          } else {
            setProblem({ kind: "generic" });
          }
        })
        .finally(() => token === request.current && setLoading(false));
    },
    [resetRun],
  );

  useEffect(() => {
    let active = true;
    loadSamples()
      .then((entries) => {
        if (!active) return;
        setSamples(entries);
        load(() => loadSampleResult(entries[0].id, "none"));
        setDatasetId(entries[0].id);
      })
      .catch(() => active && setProblem({ kind: "generic" }));
    return () => {
      active = false;
      timers.current.forEach(window.clearTimeout);
    };
  }, [load]);

  const sample = samples?.find((s) => s.id === datasetId) ?? null;
  const isUpload = datasetId === UPLOAD;

  function pickDataset(id: string) {
    if (id === datasetId) return;
    setDatasetId(id);
    setVariant("none");
    if (id === UPLOAD) {
      request.current++;
      resetRun();
      setResult(null);
      setFile(null);
      setProblem(null);
      setLoading(false);
    } else {
      load(() => loadSampleResult(id, "none"));
    }
  }

  function pickTarget(target: string | null) {
    const next = target ?? "none";
    if (next === variant) return;
    setVariant(next);
    if (isUpload && file) load(() => prepareUpload(file, target));
    else if (!isUpload) load(() => loadSampleResult(datasetId, next));
  }

  function pickFile(chosen: File | undefined) {
    if (!chosen) return;
    setFile(chosen);
    setVariant("none");
    load(() => prepareUpload(chosen, null));
  }

  function run() {
    if (!result || running) return;
    resetRun();
    setRunning(true);
    let elapsed = 0;
    result.steps.forEach((step, i) => {
      elapsed += stepDelay(step.seconds);
      timers.current.push(window.setTimeout(() => setStage(i + 1), elapsed));
    });
    timers.current.push(
      window.setTimeout(() => {
        setRunning(false);
        setRan(true);
        setView("after");
      }, elapsed + 120),
    );
  }

  const targets = isUpload
    ? (result?.dataset.target_options ?? []).slice(0, MAX_TARGET_CHIPS)
    : (sample?.targets ?? []);
  const stem = isUpload ? (file?.name.replace(/\.[^.]+$/, "") ?? "upload") : datasetId;
  const steps = result?.steps ?? [];
  const runDisabled = !result || running || loading;
  const runLabel = isUpload && !file ? t.chooseFile : running ? t.running : ran ? t.rerun : t.run;

  const stats: { key: StatKey; value: string }[] = result
    ? [
        { key: "quality", value: `${formatPercent(result.stats.quality_before, lang)} → ${formatPercent(result.stats.quality_after, lang)}` },
        { key: "standardised", value: formatNumber(result.stats.standardised, lang) },
        { key: "duplicates", value: formatNumber(result.stats.duplicates, lang) },
        { key: "filled", value: formatNumber(result.stats.filled, lang) },
      ]
    : [];

  const tabs: { id: Tab; label: string }[] = [
    { id: "data", label: t.tabData },
    { id: "why", label: t.tabWhy },
    { id: "summary", label: t.tabSummary },
    { id: "export", label: t.tabExport },
  ];

  return (
    <section id="demo" className={styles.demo}>
      <div className={styles.demoHead}>
        <div>
          <h2 className={styles.demoTitle}>{t.demoTitle}</h2>
          <p className={styles.demoSub}>{t.demoSub}</p>
        </div>
        <span className={styles.label}>{t.privacy}</span>
      </div>

      <div className={styles.panel}>
        <div role="tablist" aria-label={t.datasets} className={styles.datasets}>
          {(samples ?? []).map((entry) => (
            <button key={entry.id} type="button" role="tab" aria-selected={entry.id === datasetId}
                    className={styles.datasetTab} onClick={() => pickDataset(entry.id)}>
              <span className={styles.label}>{entry.domain[lang].toUpperCase()}</span>
              <span className={styles.datasetName}>{entry.name[lang]}</span>
            </button>
          ))}
          <button type="button" role="tab" aria-selected={isUpload}
                  className={`${styles.datasetTab} ${styles.datasetUpload}`} onClick={() => pickDataset(UPLOAD)}>
            <span className={styles.label}>{t.uploadDomain}</span>
            <span className={styles.datasetName}>{t.upload}</span>
          </button>
        </div>

        <div className={styles.body}>
          <aside className={styles.side}>
            <div className={styles.block}>
              <span className={styles.datasetTitle}>{isUpload ? (file?.name ?? t.yourFile) : sample?.name[lang]}</span>
              {isUpload ? (
                <span className={styles.facts}>
                  {result ? `${formatNumber(result.dataset.rows, lang)} × ${formatNumber(result.dataset.columns, lang)}` : t.uploadFacts}
                </span>
              ) : sample && (
                <>
                  <span className={styles.facts}>
                    {formatNumber(sample.rows, lang)} × {formatNumber(sample.columns, lang)} · {t.licence}: {sample.licence}
                  </span>
                  <span className={styles.source}>
                    {t.source}: {sample.url ? <a href={sample.url}>{sample.source[lang]}</a> : sample.source[lang]}
                  </span>
                </>
              )}
            </div>

            {isUpload && !file && <p className={styles.hint}>{t.uploadHint}</p>}

            {result && (
              <div className={styles.block}>
                <span className={styles.label}>{ran ? t.issuesFixed : t.issuesFound}</span>
                {result.issues.length === 0 ? (
                  <span className={styles.hint}>{t.noIssues}</span>
                ) : (
                  <ul className={styles.issues}>
                    {result.issues.map((issue) => (
                      <li key={issue.id} className={styles.issue} title={issue.columns.join(", ")}>
                        {ran ? <Tick /> : <span className={styles.dotWarn} aria-hidden="true" />}
                        <span className={styles.issueName}>{issue.label[lang]}</span>
                        <span className={styles.issueCount}>{formatNumber(issue.count, lang)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {(result || (isUpload && file)) && targets.length > 0 && (
              <div className={styles.block}>
                <span id="predict-label" className={styles.label}>{t.predict}</span>
                <div role="radiogroup" aria-labelledby="predict-label" className={styles.chips}>
                  {[null, ...targets].map((target) => (
                    <button key={target ?? "none"} type="button" role="radio" className={styles.chip}
                            aria-checked={(target ?? "none") === variant} onClick={() => pickTarget(target)}>
                      {target ?? t.predictNone}
                    </button>
                  ))}
                </div>
                <span className={styles.hint}>{t.predictHint}</span>
              </div>
            )}

            <button type="button" className={styles.run} disabled={runDisabled} onClick={run}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M4 2.5v11l9-5.5z" />
              </svg>
              {runLabel}
            </button>

            {steps.length > 0 && (
              <div className={styles.block}>
                <span className={styles.label}>{t.pipeline}</span>
                <ol className={styles.steps}>
                  {steps.map((step, i) => {
                    const done = ran || i < stage;
                    const active = running && i === stage;
                    return (
                      <li key={step.id} className={`${styles.step} ${done || active ? styles.stepActive : ""}`}>
                        <span className={styles.stepIcon} aria-hidden="true">
                          {done ? <Tick /> : active ? <span className={styles.stepRunning} /> : <span className={styles.stepPending} />}
                        </span>
                        <span className={styles.stepName}>{step.label[lang]}</span>
                        <span className={styles.stepTime}>{done ? formatSeconds(step.seconds, lang) : ""}</span>
                      </li>
                    );
                  })}
                </ol>
              </div>
            )}
          </aside>

          <div className={styles.main}>
            {problem && (
              <div role="alert" className={styles.message}>
                {problem.kind === "unavailable" ? t.apiUnavailable : problem.kind === "message" ? problem.text : t.error}
              </div>
            )}

            {isUpload && !file && (
              <label htmlFor="file-upload"
                     className={`${styles.drop} ${dragging ? styles.dropActive : ""}`}
                     onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                     onDragLeave={() => setDragging(false)}
                     onDrop={(e) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}>
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                  <path d="M20 27V9M13 16l7-7 7 7M8 31h24" />
                </svg>
                <span className={styles.dropTitle}>{t.dropTitle}</span>
                <span className={styles.hint}>{t.dropSub}</span>
                <input id="file-upload" type="file" accept=".csv,.xlsx,.xls"
                       onChange={(e) => pickFile(e.target.files?.[0])} />
              </label>
            )}

            {loading && <div className={styles.loading}>{isUpload ? t.uploading : t.loading}</div>}

            {!loading && result && (
              <>
                {ran && (
                  <div className={styles.stats}>
                    {stats.map((stat) => (
                      <div key={stat.key} className={styles.stat}>
                        <span className={styles.statLabel}>{result.stat_labels[stat.key][lang]}</span>
                        <span className={styles.statValue}>{stat.value}</span>
                      </div>
                    ))}
                  </div>
                )}

                <div role="tablist" aria-label={t.results} className={styles.tabs}>
                  {tabs.map(({ id, label }) => {
                    const disabled = id !== "data" && !ran;
                    return (
                      <button key={id} type="button" role="tab" className={styles.tab}
                              aria-selected={tab === id} aria-disabled={disabled}
                              onClick={() => !disabled && setTab(id)}>
                        {label}
                      </button>
                    );
                  })}
                </div>

                {tab === "data" && (
                  <Preview result={result} ran={ran} view={view} onView={setView}
                           selected={selected} onSelect={setSelected} />
                )}
                {tab === "why" && <Decisions result={result} />}
                {tab === "summary" && <Summary result={result} />}
                {tab === "export" && <Export result={result} fromSample={!isUpload} stem={stem} />}
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

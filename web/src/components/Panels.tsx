"use client";

import { useEffect, useState } from "react";
import { downloadText, saveDownload } from "@/lib/data";
import { useLang } from "@/lib/lang";
import type { DownloadKey, Result } from "@/lib/types";
import styles from "./Workspace.module.css";

export function Decisions({ result }: { result: Result }) {
  const { lang } = useLang();
  return (
    <div className={styles.cards}>
      {result.decisions.map((card, i) => (
        <div key={i} className={styles.card}>
          <span className={styles.cardTag}>{card.tag[lang]}</span>
          <span className={styles.cardTitle}>{card.title[lang]}</span>
          <span className={styles.cardBody}>{card.body[lang]}</span>
        </div>
      ))}
    </div>
  );
}

export function Summary({ result }: { result: Result }) {
  const { lang, t } = useLang();
  return (
    <div className={styles.summary}>
      <span className={styles.badge}>{t.summaryBadge}</span>
      <p className={styles.summaryText}>{result.summary[lang]}</p>
      <span className={styles.note}>{t.summaryNote}</span>
    </div>
  );
}

const DOWNLOADS: { key: DownloadKey; suffix: string }[] = [
  { key: "clean_csv", suffix: "clean.csv" },
  { key: "ml_csv", suffix: "ml-ready.csv" },
  { key: "shopify_csv", suffix: "shopify.csv" },
  { key: "report_json", suffix: "report.json" },
];

export function Export({ result, fromSample, stem }: { result: Result; fromSample: boolean; stem: string }) {
  const { t } = useLang();
  const [shopifyLines, setShopifyLines] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    downloadText(result, "shopify_csv", fromSample)
      .then((text) => current && setShopifyLines(text ? text.split("\n").slice(0, 5).join("\n") : null))
      .catch(() => current && setShopifyLines(null));
    return () => {
      current = false;
    };
  }, [result, fromSample]);

  const labels: Record<DownloadKey, [string, string]> = {
    clean_csv: [t.dlClean, t.dlCleanNote],
    ml_csv: [t.dlMl, t.dlMlNote],
    shopify_csv: [t.dlShopify, result.downloads.shopify_csv ? t.dlShopifyNote : t.dlShopifyOff],
    report_json: [t.dlReport, t.dlReportNote],
  };

  return (
    <div className={styles.downloads}>
      {DOWNLOADS.map(({ key, suffix }) => {
        const [label, note] = labels[key];
        const available = result.downloads[key] !== null;
        return (
          <button key={key} type="button" className={styles.download} disabled={!available}
                  onClick={() => saveDownload(result, key, fromSample, `${stem}-${suffix}`)}>
            <span className={styles.downloadText}>
              <span className={styles.downloadLabel}>{label}</span>
              <span className={styles.downloadNote}>{note}</span>
            </span>
            <svg width="18" height="18" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path d="M8 2v9M4 7l4 4 4-4M2.5 14h11" />
            </svg>
          </button>
        );
      })}
      {shopifyLines && (
        <div className={styles.block}>
          <span className={styles.label}>{t.shopifyPreview}</span>
          <pre className={styles.code}>{shopifyLines}</pre>
        </div>
      )}
    </div>
  );
}

"use client";

import type { ReactNode } from "react";
import { useLang } from "@/lib/lang";
import type { Cell, Result } from "@/lib/types";
import styles from "./Workspace.module.css";

export type View = "before" | "after";
export type CellPosition = { row: number; column: number };

interface Props {
  result: Result;
  ran: boolean;
  view: View;
  onView: (view: View) => void;
  selected: CellPosition | null;
  onSelect: (position: CellPosition) => void;
}

/** Shows a value as written, with leading/trailing spaces made visible ("navy␣"). */
function Value({ value, emptyLabel }: { value: string | null; emptyLabel: string }) {
  if (value === null) return <span className={styles.empty}>{emptyLabel}</span>;
  const [, lead, core, trail] = value.match(/^(\s*)([\s\S]*?)(\s*)$/) ?? ["", "", value, ""];
  return (
    <>
      {lead && <span className={styles.space}>{"␣".repeat(lead.length)}</span>}
      {core}
      {trail && <span className={styles.space}>{"␣".repeat(trail.length)}</span>}
    </>
  );
}

function cellClass(cell: Cell, showAfter: boolean, isSelected: boolean) {
  const kind = {
    ok: "",
    note: styles.cellNote,
    fixed: showAfter ? styles.cellFixed : styles.cellBad,
    removed: showAfter ? styles.cellRemoved : styles.cellBad,
  }[cell.kind];
  return [styles.cell, kind, isSelected ? styles.cellSelected : ""].filter(Boolean).join(" ");
}

export function Preview({ result, ran, view, onView, selected, onSelect }: Props) {
  const { lang, t } = useLang();
  const { columns, rows } = result.preview;
  const showAfter = ran && view === "after";
  // Column widths follow their longest value, so a fix at the end of a long text stays visible
  const widths = columns.map((column, c) => {
    const longest = Math.max(column.length, ...rows.flatMap((row) => [row.cells[c].before, row.cells[c].after]).map((v) => (v ?? t.empty).length));
    return Math.min(300, Math.max(72, Math.round(longest * 8.6 + 24)));
  });
  const template = ["64px", ...widths.map((w) => `minmax(${w}px, ${w}fr)`)].join(" ");
  const minWidth = 64 + widths.reduce((sum, w) => sum + w, 0);

  return (
    <div className={styles.block}>
      <div className={styles.toolbar}>
        <div role="group" aria-label={t.view} className={styles.segmented}>
          <button type="button" className={styles.segment} aria-pressed={!showAfter} onClick={() => onView("before")}>
            {t.before}
          </button>
          <button type="button" className={styles.segment} aria-pressed={showAfter} disabled={!ran} onClick={() => onView("after")}>
            {t.after}
          </button>
        </div>
        <div className={styles.legend}>
          <span className={styles.legendItem}><span className={styles.swatchBad} />{t.legendBad}</span>
          <span className={styles.legendItem}><span className={styles.swatchFixed} />{t.legendFixed}</span>
          <span className={`${styles.legendItem} ${styles.legendRemoved}`}>{t.legendRemoved}</span>
        </div>
      </div>

      <div className={styles.tableBox}>
        <div role="table" aria-label={`${result.dataset.name[lang]} · ${showAfter ? t.after : t.before}`}
             className={styles.table} style={{ minWidth }}>
          <div role="row" className={`${styles.tableRow} ${styles.tableHead}`} style={{ gridTemplateColumns: template }}>
            <span role="columnheader" className={styles.headCell}>{t.row}</span>
            {columns.map((column) => (
              <span key={column} role="columnheader" className={styles.headCell} title={column}>{column}</span>
            ))}
          </div>
          {rows.map((row, r) => (
            <div key={row.row} role="row" className={styles.tableRow} style={{ gridTemplateColumns: template }}>
              <span role="rowheader" className={styles.rowNumber}>{row.row}</span>
              {row.cells.map((cell, c) => {
                const value = showAfter && cell.kind !== "removed" ? cell.after : cell.before;
                const isSelected = selected?.row === r && selected?.column === c;
                const content = <Value value={value} emptyLabel={t.empty} />;
                return (
                  <span key={columns[c]} role="cell" style={{ minWidth: 0 }}>
                    {cell.reason ? (
                      <button type="button" className={cellClass(cell, showAfter, isSelected)}
                              aria-label={`${columns[c]}: ${value ?? t.empty}`} title={value ?? undefined}
                              onClick={() => onSelect({ row: r, column: c })}>
                        {content}
                      </button>
                    ) : (
                      <span className={cellClass(cell, showAfter, false)} title={value ?? undefined}>{content}</span>
                    )}
                  </span>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <Inspector result={result} ran={ran} selected={selected} />
      <span className={styles.note}>{t.previewNote}</span>
    </div>
  );
}

function Inspector({ result, ran, selected }: { result: Result; ran: boolean; selected: CellPosition | null }) {
  const { lang, t } = useLang();
  const cell = selected ? result.preview.rows[selected.row]?.cells[selected.column] : null;
  const reason = cell?.reason ? result.preview.reasons[cell.reason]?.[lang] : null;

  let tone = "";
  let change: ReactNode = null;
  let text: ReactNode = t.inspectHint;
  if (cell && reason) {
    const changed = cell.kind === "fixed" || cell.kind === "removed";
    if (changed && ran) {
      tone = styles.inspectorFixed;
      change = (
        <span className={styles.inspectorChange}>
          {result.preview.columns[selected!.column]} ·{" "}
          <span className={styles.from}><Value value={cell.before} emptyLabel={t.empty} /></span>
          {" → "}
          <span className={styles.to}>{cell.kind === "removed" ? t.legendRemoved : <Value value={cell.after} emptyLabel={t.empty} />}</span>
        </span>
      );
      text = reason;
    } else if (changed) {
      tone = styles.inspectorWarn;
      text = t.willFix + reason;
    } else {
      text = reason;
    }
  }

  return (
    <div aria-live="polite" className={`${styles.inspector} ${tone}`}>
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"
           style={{ flexShrink: 0, marginTop: 2 }}>
        <circle cx="9" cy="9" r="7.2" />
        <path d="M9 8v5M9 5.2v.1" />
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {change}
        <span>{text}</span>
      </div>
    </div>
  );
}

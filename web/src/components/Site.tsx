"use client";

import { useEffect, useState } from "react";
import { loadBenchmark } from "@/lib/data";
import { formatNumber } from "@/lib/i18n";
import { useLang } from "@/lib/lang";
import type { Benchmark as BenchmarkData } from "@/lib/types";
import styles from "./Site.module.css";

export const LINKS = {
  portfolio: "https://parinaz-teimouri.vercel.app/",
  github: "https://github.com/parinaz-teimouri",
  linkedin: "https://www.linkedin.com/in/parinaz-teimouri/",
  email: "parinaztmr@gmail.com",
  benchmark:
    process.env.NEXT_PUBLIC_BENCHMARK_URL ??
    "https://github.com/parinaz-teimouri/Intelligent-Automation-Data-Preparation/blob/main/benchmark_results/RESULTS.md",
};

export function Header() {
  const { lang, t, setLang } = useLang();
  return (
    <header className={styles.header}>
      <div className={styles.headerInner}>
        <a href={LINKS.portfolio} className={styles.brand}>
          PARI / data-prep
        </a>
        <nav aria-label="Main" className={styles.nav}>
          <a href="#demo" className={styles.navLink}>{t.navTry}</a>
          <a href="#how" className={styles.navLink}>{t.navHow}</a>
          <a href="#bench" className={styles.navLink}>{t.navBench}</a>
          <a href={LINKS.github} className={styles.navLink}>GitHub</a>
          <div role="group" aria-label="Language / Sprache" className={styles.langGroup}>
            <button type="button" className={styles.langButton} aria-pressed={lang === "en"} onClick={() => setLang("en")}>
              EN
            </button>
            <button type="button" className={styles.langButton} aria-pressed={lang === "de"} onClick={() => setLang("de")}>
              DE
            </button>
          </div>
        </nav>
      </div>
    </header>
  );
}

export function Hero() {
  const { t } = useLang();
  return (
    <section className={styles.hero}>
      <div className={styles.heroText}>
        <p className={styles.kicker}>{t.kicker}</p>
        <h1 className={styles.heroTitle}>{t.heroTitle}</h1>
        <p className={styles.heroSub}>{t.heroSub}</p>
        <div className={styles.ctas}>
          <a href="#demo" className={styles.buttonPrimary}>{t.ctaTry}</a>
          <a href="#how" className={styles.buttonSecondary}>{t.ctaHow}</a>
        </div>
      </div>
      <div className={styles.heroVisual} aria-hidden="true">
        <div className={styles.visualColumn}>
          <span className={styles.visualLabel}>RAW</span>
          <span className={styles.rawToken}>&quot;NAVY &quot;</span>
          <span className={styles.rawToken}>&quot;89,90 €&quot;</span>
          <span className={styles.rawToken}>&quot;Ja&quot;</span>
          <span className={styles.rawToken}>NULL</span>
        </div>
        <div className={styles.visualSteps}>
          <span>trim</span>
          <span>cast</span>
          <span>merge</span>
          <span>fill</span>
          <svg width="28" height="14" viewBox="0 0 28 14" fill="none" stroke="var(--accent)" strokeWidth="1.8" style={{ margin: "4px auto 0" }}>
            <path d="M1 7h24M19 1l6 6-6 6" />
          </svg>
        </div>
        <div className={styles.visualColumn}>
          <span className={styles.visualLabel}>TYPED</span>
          <span className={styles.typedToken}>colour: Navy</span>
          <span className={styles.typedToken}>price: 89.90</span>
          <span className={styles.typedToken}>on_sale: 1</span>
          <span className={styles.typedToken}>stock: 14</span>
        </div>
      </div>
    </section>
  );
}

export function HowItWorks() {
  const { t } = useLang();
  return (
    <section id="how" className={styles.section}>
      <h2 className={styles.sectionTitle}>{t.howTitle}</h2>
      <div className={styles.howGrid}>
        {t.how.map((step, i) => (
          <div key={step.title} className={styles.howItem}>
            <span className={styles.howNumber}>{String(i + 1).padStart(2, "0")}</span>
            <span className={styles.howTitle}>{step.title}</span>
            <span className={styles.howBody}>{step.body}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function Benchmark() {
  const { lang, t } = useLang();
  const [data, setData] = useState<BenchmarkData | null>(null);

  useEffect(() => {
    loadBenchmark().then(setData).catch(() => setData(null));
  }, []);

  if (!data) return <section id="bench" className={styles.bench} />;
  const cleaning = data.cleaning_vs_static;
  const selection = data.selection_vs_cleaning;
  const total = cleaning.wins + cleaning.ties + cleaning.losses;
  const points = (d: number) => `+${formatNumber(d * 100, lang, 1)} ${t.points}`;

  return (
    <section id="bench" className={styles.bench}>
      <div className={styles.benchCard}>
        <h2 className={styles.sectionTitle}>{t.benchTitle}</h2>
        <p className={styles.benchBody}>{t.benchBody(data.datasets.length, data.repeats)}</p>
        <div className={styles.benchGrid}>
          <div className={styles.benchItem}>
            <span className={styles.benchValue}>{points(cleaning.mean_difference)}</span>
            <span>{t.benchCleaning(cleaning.wins, total)}</span>
          </div>
          <div className={styles.benchItem}>
            <span className={styles.benchValue}>{points(selection.mean_difference)}</span>
            <span>{t.benchSelection}</span>
          </div>
        </div>
        <a href={LINKS.benchmark} className={styles.benchLink}>{t.benchLink}</a>
      </div>
    </section>
  );
}

export function Footer() {
  const { t } = useLang();
  return (
    <footer className={styles.footer}>
      <div className={styles.footerInner}>
        <div className={styles.footerRow}>
          <span>{t.footer}</span>
          <nav aria-label="Contact" className={styles.footerNav}>
            <a href={LINKS.portfolio}>Portfolio</a>
            <a href={LINKS.github}>GitHub</a>
            <a href={LINKS.linkedin}>LinkedIn</a>
            <a href={`mailto:${LINKS.email}`}>{LINKS.email}</a>
          </nav>
        </div>
        <p className={styles.credits}>{t.credits}</p>
      </div>
    </footer>
  );
}

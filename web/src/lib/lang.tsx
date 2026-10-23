"use client";

import { createContext, useCallback, useContext, useEffect, useSyncExternalStore } from "react";
import { UI, type Copy } from "./i18n";
import type { Lang } from "./types";

const STORAGE_KEY = "lang";
const listeners = new Set<() => void>();
let chosen: Lang | null = null; // the toggle wins, even when storage is blocked

function readLang(): Lang {
  if (chosen) return chosen;
  try {
    const fromUrl = new URLSearchParams(window.location.search).get("lang");
    if (fromUrl === "de" || fromUrl === "en") return fromUrl;
    return window.localStorage.getItem(STORAGE_KEY) === "de" ? "de" : "en";
  } catch {
    return "en";
  }
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const LangContext = createContext<{ lang: Lang; t: Copy; setLang: (lang: Lang) => void }>({
  lang: "en",
  t: UI.en,
  setLang: () => {},
});

export function LangProvider({ children }: { children: React.ReactNode }) {
  // English while pre-rendering; the visitor's choice (?lang=de or a stored one) after hydration
  const lang = useSyncExternalStore(subscribe, readLang, () => "en" as Lang);

  const setLang = useCallback((next: Lang) => {
    chosen = next;
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
      const url = new URL(window.location.href);
      url.searchParams.delete("lang");
      window.history.replaceState(null, "", url);
    } catch {
      // Storage can be blocked; the choice then lasts until the page is reloaded
    }
    listeners.forEach((listener) => listener());
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  return <LangContext.Provider value={{ lang, t: UI[lang], setLang }}>{children}</LangContext.Provider>;
}

export const useLang = () => useContext(LangContext);

"""
Plain-language explanations of a pipeline run, in English and German.

Everything here is written from the facts collected by api.results (what the pipeline logged),
so every sentence corresponds to a decision that was actually made.
"""

LANGS = ("en", "de")

STRATEGY = {
    "imputation": {
        "mean": {"en": "the mean", "de": "dem Mittelwert"},
        "median": {"en": "the median", "de": "dem Median"},
        "knn": {"en": "values from the most similar rows (KNN)", "de": "Werten der ähnlichsten Zeilen (KNN)"},
        "most_frequent": {"en": "the most common value", "de": "dem häufigsten Wert"},
    },
    "imputation_short": {
        "mean": {"en": "mean", "de": "Mittelwert"},
        "median": {"en": "median", "de": "Median"},
        "knn": {"en": "KNN", "de": "KNN"},
        "most_frequent": {"en": "most common value", "de": "häufigster Wert"},
    },
    "scaling": {
        "standard": {"en": "standard scaling (mean 0, spread 1)", "de": "Standardisierung (Mittelwert 0, Streuung 1)"},
        "robust": {"en": "robust scaling (median and quartiles)", "de": "robuste Skalierung (Median und Quartile)"},
        "minmax": {"en": "min-max scaling (0 to 1)", "de": "Min-Max-Skalierung (0 bis 1)"},
    },
    "scaling_short": {
        "standard": {"en": "standard", "de": "Standard"},
        "robust": {"en": "robust", "de": "robust"},
        "minmax": {"en": "min-max", "de": "Min-Max"},
    },
}

ISSUES = {
    "spelling": {"en": "Spelling variants", "de": "Schreibvarianten"},
    "numbers_text": {"en": "Numbers stored as text", "de": "Zahlen als Text"},
    "yes_no_text": {"en": "Yes/no as text", "de": "Ja/Nein als Text"},
    "dates_text": {"en": "Dates stored as text", "de": "Datum als Text"},
    "missing": {"en": "Missing values", "de": "Fehlende Werte"},
    "duplicates": {"en": "Duplicate rows", "de": "Doppelte Zeilen"},
    "identifiers": {"en": "ID or free-text columns", "de": "ID- oder Freitextspalten"},
    "missing_target": {"en": "Rows without a target value", "de": "Zeilen ohne Zielwert"},
    "outliers": {"en": "Extreme values", "de": "Extremwerte"},
}

STATS = {
    "quality": {"en": "Data quality", "de": "Datenqualität"},
    "standardised": {"en": "Values standardised", "de": "Werte vereinheitlicht"},
    "duplicates": {"en": "Duplicates removed", "de": "Duplikate entfernt"},
    "filled": {"en": "Gaps filled", "de": "Lücken gefüllt"},
}

STEPS = {
    "load": {"en": "Load", "de": "Laden"},
    "clean": {"en": "Clean values", "de": "Werte bereinigen"},
    "duplicates": {"en": "Remove duplicates", "de": "Duplikate entfernen"},
    "impute": {"en": "Fill gaps", "de": "Lücken füllen"},
    "encode": {"en": "Encode categories", "de": "Kategorien kodieren"},
    "scale": {"en": "Scale numbers", "de": "Zahlen skalieren"},
    "report": {"en": "Explain", "de": "Erklären"},
}


def num(value, lang):
    """1234 -> '1,234' (en) / '1.234' (de)."""
    text = f"{int(round(value)):,}"
    return text.replace(",", ".") if lang == "de" else text


def dec(value, lang, digits=2):
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "de" else text


def quote(value, lang):
    return f"„{value}“" if lang == "de" else f"\"{value}\""


def listing(items, lang, limit=4):
    items = [str(item) for item in items]
    shown = items[:limit]
    more = len(items) - len(shown)
    if more:
        shown.append(f"{more} more" if lang == "en" else f"{more} weitere")
    if len(shown) == 1:
        return shown[0]
    joiner = " and " if lang == "en" else " und "
    return ", ".join(shown[:-1]) + joiner + shown[-1]


def both(render):
    """Calls render(lang) for every language: {"en": ..., "de": ...}."""
    return {lang: render(lang) for lang in LANGS}


# --- Cell-level reasons (the inspector under the table) ----------------------------

def reason_text(action, facts, extra=None):
    extra = extra or {}
    target = facts.get("target")
    imputation = facts.get("imputation") or {}

    def render(lang):
        en = lang == "en"
        if action == "merge":
            return ("The same value was written in different ways; merged into the most common spelling." if en else
                    "Derselbe Wert war unterschiedlich geschrieben; zur häufigsten Schreibweise zusammengeführt.")
        if action == "number":
            return ("Number stored as text (currency sign, percent or German notation); converted to a number." if en else
                    "Zahl als Text (Währungszeichen, Prozent oder deutsche Schreibweise); in eine Zahl umgewandelt.")
        if action == "boolean":
            return ("Yes/no written as text; converted to 1 or 0." if en else
                    "Ja/Nein als Text; in 1 oder 0 umgewandelt.")
        if action == "date":
            return ("Date stored as text; read as a real date and split into year, month and day for analysis." if en else
                    "Datum als Text; als echtes Datum eingelesen und für die Auswertung in Jahr, Monat und Tag zerlegt.")
        if action == "capped":
            return ("Rare extreme value; capped to the normal range of this column." if en else
                    "Seltener Extremwert; auf den üblichen Bereich der Spalte begrenzt.")
        if action == "fill_numeric":
            strategy = STRATEGY["imputation"].get(imputation.get("strategy"), {}).get(lang, imputation.get("strategy"))
            if imputation.get("evaluated"):
                return (f"Missing; filled with {strategy}, which scored best in a quick test against {quote(target, lang)}." if en else
                        f"Fehlte; mit {strategy} ergänzt, was im Schnelltest gegen {quote(target, lang)} am besten abschnitt.")
            return f"Missing; filled with {strategy}." if en else f"Fehlte; mit {strategy} ergänzt."
        if action == "fill_binary":
            return ("Missing yes/no value; filled with the most common answer, not an average like 0.45." if en else
                    "Fehlender Ja/Nein-Wert; mit der häufigsten Antwort ergänzt, nicht mit einem Durchschnitt wie 0,45.")
        if action == "fill_frequent":
            return ("Missing; only a few values in this column were missing, so the most common value was used." if en else
                    "Fehlte; in dieser Spalte fehlten nur wenige Werte, daher wurde der häufigste Wert verwendet.")
        if action == "fill_missing_category":
            return ("Missing in many rows; marked as its own category \"(missing)\" instead of guessing a value." if en else
                    "Fehlt in vielen Zeilen; als eigene Kategorie „(missing)“ markiert, statt einen Wert zu raten.")
        if action == "identifier":
            return ("Identifies a row but predicts nothing; kept in the clean file, left out of the analysis." if en else
                    "Kennzeichnet eine Zeile, sagt aber nichts vorher; bleibt in der bereinigten Datei, wird nicht ausgewertet.")
        if action == "identifier_missing":
            return ("Missing ID left empty: an invented ID would be worse than none." if en else
                    "Fehlende Kennung bleibt leer: Eine erfundene Kennung wäre schlimmer als keine.")
        if action == "duplicate":
            row = extra.get("of", 0) + 1
            return (f"Exact copy of row {row}; removed." if en else f"Exakte Kopie von Zeile {row}; entfernt.")
        if action == "missing_target":
            return (f"No value for {quote(target, lang)}, so the row cannot be used to test strategies; removed." if en else
                    f"Kein Wert für {quote(target, lang)}, daher kann die Zeile nicht zum Testen genutzt werden; entfernt.")
        if action == "unreadable_target":
            return (f"The {quote(target, lang)} value could not be read as a number; row removed." if en else
                    f"Der Wert für {quote(target, lang)} ließ sich nicht als Zahl lesen; Zeile entfernt.")
        return "Changed during cleaning." if en else "Beim Bereinigen geändert."

    return both(render)


# --- Decision cards ("Why did it do that?") -------------------------------------------

def _card(tag, title, body):
    return {"tag": tag, "title": title, "body": body}


def decision_cards(facts):
    cards = []
    metric = facts.get("metric", "F1")

    for merge in facts["merges"]:
        column = merge["column"]
        examples = merge["examples"]
        cards.append(_card(
            {"en": column.upper(), "de": column.upper()},
            both(lambda l: "Merged spelling variants" if l == "en" else "Schreibvarianten zusammengeführt"),
            both(lambda l: (
                f"{listing([quote(a, l) for a, _ in examples], l, 3)} now read {listing(sorted({quote(b, l) for _, b in examples}), l, 3)}, "
                f"the most common spelling ({num(merge['cells'], l)} values)." if l == "en" else
                f"{listing([quote(a, l) for a, _ in examples], l, 3)} lauten jetzt {listing(sorted({quote(b, l) for _, b in examples}), l, 3)}, "
                f"die häufigste Schreibweise ({num(merge['cells'], l)} Werte).")),
        ))

    for number in facts["numbers"]:
        raw, parsed = number["example"]
        cards.append(_card(
            {"en": number["column"].upper(), "de": number["column"].upper()},
            both(lambda l: "Converted text to numbers" if l == "en" else "Text in Zahlen umgewandelt"),
            both(lambda l: (
                f"{num(number['cells'], l)} values like {quote(raw, l)} are now numbers ({parsed}). "
                f"Currency signs, percent signs and German notation are recognised." if l == "en" else
                f"{num(number['cells'], l)} Werte wie {quote(raw, l)} sind jetzt Zahlen ({parsed.replace('.', ',')}). "
                f"Währungs- und Prozentzeichen sowie die deutsche Schreibweise werden erkannt.")),
        ))

    for flag in facts["booleans"]:
        cards.append(_card(
            {"en": flag["column"].upper(), "de": flag["column"].upper()},
            both(lambda l: "Turned yes/no text into 1/0" if l == "en" else "Ja/Nein-Text in 1/0 umgewandelt"),
            both(lambda l: (
                f"{listing([quote(v, l) for v in flag['values']], l, 6)} become 1 or 0, so the column can be counted and filtered." if l == "en" else
                f"{listing([quote(v, l) for v in flag['values']], l, 6)} werden zu 1 oder 0, damit die Spalte gezählt und gefiltert werden kann.")),
        ))

    for date in facts["dates"]:
        if date.get("fixed"):
            title = {"en": "Read text dates", "de": "Text-Datumswerte eingelesen"}
            body = both(lambda l: (
                f"Values like {quote(date['example'], l)} are real dates now, and year, month and day were added as columns for analysis." if l == "en" else
                f"Werte wie {quote(date['example'], l)} sind jetzt echte Daten; Jahr, Monat und Tag wurden als Spalten für die Auswertung ergänzt."))
        else:
            title = {"en": "Split dates for analysis", "de": "Datum für die Auswertung zerlegt"}
            body = both(lambda l: (
                "The dates were already in a standard format; year, month and day were added as columns so models can use them." if l == "en" else
                "Die Daten waren bereits im Standardformat; Jahr, Monat und Tag wurden als Spalten ergänzt, damit Modelle sie nutzen können."))
        cards.append(_card({"en": date["column"].upper(), "de": date["column"].upper()}, title, body))

    if facts["duplicates"]:
        cards.append(_card(
            {"en": "ROWS", "de": "ZEILEN"},
            both(lambda l: "Removed duplicate rows" if l == "en" else "Doppelte Zeilen entfernt"),
            both(lambda l: (
                f"{num(facts['duplicates'], l)} rows were exact copies of another row; the copies were removed." if l == "en" else
                f"{num(facts['duplicates'], l)} Zeilen waren exakte Kopien einer anderen Zeile; die Kopien wurden entfernt.")),
        ))

    if facts["missing_target"] or facts["unreadable_target"]:
        dropped = facts["missing_target"] + facts["unreadable_target"]
        cards.append(_card(
            {"en": "TARGET", "de": "ZIEL"},
            both(lambda l: "Removed rows without a target" if l == "en" else "Zeilen ohne Zielwert entfernt"),
            both(lambda l: (
                f"{num(dropped, l)} rows had no usable {quote(facts['target'], l)} value. Inventing one would make the tests below meaningless." if l == "en" else
                f"{num(dropped, l)} Zeilen hatten keinen verwertbaren Wert für {quote(facts['target'], l)}. Einen zu erfinden würde die Tests unten wertlos machen.")),
        ))

    if facts["identifiers"]:
        cards.append(_card(
            {"en": "IDS", "de": "KENNUNGEN"},
            both(lambda l: "Kept identifiers out of the analysis" if l == "en" else "Kennungen nicht ausgewertet"),
            both(lambda l: (
                f"{listing(facts['identifiers'], l)} identify rows but predict nothing. They stay in the clean file." if l == "en" else
                f"{listing(facts['identifiers'], l)} kennzeichnen Zeilen, sagen aber nichts vorher. Sie bleiben in der bereinigten Datei.")),
        ))

    for capped in facts["capped"]:
        cards.append(_card(
            {"en": capped["column"].upper(), "de": capped["column"].upper()},
            both(lambda l: "Capped rare extreme values" if l == "en" else "Seltene Extremwerte begrenzt"),
            both(lambda l: (
                f"{num(capped['count'], l)} values far outside the usual range were capped in the ML-ready file, so they "
                f"cannot dominate a model. The clean file keeps the real values." if l == "en" else
                f"{num(capped['count'], l)} Werte weit außerhalb des üblichen Bereichs wurden in der ML-Datei begrenzt, damit "
                f"sie kein Modell dominieren. Die bereinigte Datei behält die echten Werte.")),
        ))

    imputation = facts.get("imputation")
    if imputation:
        cards.append(_card(
            {"en": "GAPS", "de": "LÜCKEN"},
            both(lambda l: "Filled missing numbers" if l == "en" else "Fehlende Zahlen ergänzt"),
            both(lambda l: _tested_sentence("imputation", imputation, facts, metric, l)),
        ))
    for kind, title_en, title_de, body_en, body_de in [
        ("missing_category", "Kept large gaps visible", "Große Lücken sichtbar gelassen",
         "{cols}: so many values were missing that a guess would mislead; they are marked \"(missing)\".",
         "{cols}: So viele Werte fehlten, dass eine Vermutung irreführen würde; sie sind als „(missing)“ markiert."),
        ("frequent_fill", "Filled a few missing categories", "Wenige fehlende Kategorien ergänzt",
         "{cols}: only a few values were missing, so the most common value is a safe choice.",
         "{cols}: Nur wenige Werte fehlten, daher ist der häufigste Wert eine sichere Wahl."),
        ("binary_fill", "Filled missing yes/no values", "Fehlende Ja/Nein-Werte ergänzt",
         "{cols}: filled with the most common answer, so every value stays 0 or 1.",
         "{cols}: mit der häufigsten Antwort ergänzt, damit jeder Wert 0 oder 1 bleibt."),
    ]:
        group = facts.get(kind)
        if group and group["columns"]:
            cards.append(_card(
                {"en": "GAPS", "de": "LÜCKEN"},
                {"en": title_en, "de": title_de},
                both(lambda l: (body_en if l == "en" else body_de).format(cols=listing(group["columns"], l))),
            ))

    encoding = facts.get("encoding") or {}
    if encoding.get("onehot") or encoding.get("label"):
        cards.append(_card(
            {"en": "CATEGORIES", "de": "KATEGORIEN"},
            both(lambda l: "Turned categories into numbers" if l == "en" else "Kategorien in Zahlen umgewandelt"),
            both(lambda l: _encoding_sentence(encoding, l)),
        ))

    scaling = facts.get("scaling")
    if scaling:
        cards.append(_card(
            {"en": "SCALE", "de": "SKALIERUNG"},
            both(lambda l: "Put numbers on a common scale" if l == "en" else "Zahlen auf eine gemeinsame Skala gebracht"),
            both(lambda l: _tested_sentence("scaling", scaling, facts, metric, l)),
        ))
    return cards


def _tested_sentence(task, info, facts, metric, lang):
    en = lang == "en"
    strategy = STRATEGY[task][info["strategy"]][lang]
    columns = info.get("columns", 0)
    what = ((f"Gaps in {num(columns, lang)} numeric column{'s' if columns != 1 else ''} were filled with {strategy}." if en else
             f"Lücken in {num(columns, lang)} Zahlenspalte{'n' if columns != 1 else ''} wurden mit {strategy} gefüllt.")
            if task == "imputation" else
            (f"{num(columns, lang)} numeric column{'s' if columns != 1 else ''} got {strategy}, so no column dominates a model." if en else
             f"{num(columns, lang)} Zahlenspalte{'n' if columns != 1 else ''} erhielt{'en' if columns != 1 else ''} eine {strategy}, damit keine Spalte ein Modell dominiert."))
    scores = {s: v for s, v in (info.get("scores") or {}).items() if v is not None}
    if info.get("evaluated") and len(scores) > 1:
        short = STRATEGY[f"{task}_short"]
        tested = listing([f"{short[s][lang]} {dec(v, lang)}" for s, v in scores.items()], lang)
        return what + (f" Tested against {quote(facts['target'], lang)} with a quick model ({metric}): {tested}." if en else
                       f" Gegen {quote(facts['target'], lang)} mit einem schnellen Modell getestet ({metric}): {tested}.")
    return what + (" Chosen from the data's statistics; pick a target column to test the options instead." if en else
                   " Anhand der Statistik der Daten gewählt; mit einer Zielspalte werden die Optionen stattdessen getestet.")


def _encoding_sentence(encoding, lang):
    parts = []
    if encoding.get("onehot"):
        parts.append((f"{listing(encoding['onehot'], lang)} became yes/no columns (one per value)" if lang == "en" else
                      f"{listing(encoding['onehot'], lang)} wurden zu Ja/Nein-Spalten (eine pro Wert)"))
    if encoding.get("label"):
        parts.append((f"{listing(encoding['label'], lang)} became numbered codes, because they have many values" if lang == "en" else
                      f"{listing(encoding['label'], lang)} wurden zu Nummerncodes, weil sie viele Werte haben"))
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + ". This only affects the ML-ready file." if lang == "en" else \
        sentence[0].upper() + sentence[1:] + ". Das betrifft nur die ML-Datei."


# --- Automatic summary ---------------------------------------------------------------------

def summary(facts, stats, name):
    def render(lang):
        en = lang == "en"
        sentences = [
            (f"The dataset {quote(name[lang], lang)} had {num(facts['rows_in'], lang)} rows and {num(facts['cols_in'], lang)} columns." if en else
             f"Der Datensatz {quote(name[lang], lang)} enthielt {num(facts['rows_in'], lang)} Zeilen und {num(facts['cols_in'], lang)} Spalten.")
        ]
        fixes = []
        if facts["merges"]:
            fixes.append((f"spelling variants in {listing([m['column'] for m in facts['merges']], lang)} were merged" if en else
                          f"Schreibvarianten in {listing([m['column'] for m in facts['merges']], lang)} wurden zusammengeführt"))
        if facts["numbers"]:
            fixes.append((f"numbers stored as text in {listing([n['column'] for n in facts['numbers']], lang)} became real numbers" if en else
                          f"Zahlen als Text in {listing([n['column'] for n in facts['numbers']], lang)} wurden zu echten Zahlen"))
        if facts["booleans"]:
            fixes.append((f"yes/no text in {listing([b['column'] for b in facts['booleans']], lang)} became 1/0" if en else
                          f"Ja/Nein-Text in {listing([b['column'] for b in facts['booleans']], lang)} wurde zu 1/0"))
        text_dates = [d["column"] for d in facts["dates"] if d.get("fixed")]
        if text_dates:
            fixes.append((f"text dates in {listing(text_dates, lang)} became real dates" if en else
                          f"Text-Datumswerte in {listing(text_dates, lang)} wurden zu echten Daten"))
        if fixes:
            sentences.append((f"{num(stats['standardised'], lang)} values were written inconsistently and fixed: " if en else
                              f"{num(stats['standardised'], lang)} Werte waren uneinheitlich erfasst und wurden bereinigt: ")
                             + "; ".join(fixes) + ".")
        if facts["duplicates"]:
            sentences.append((f"{num(facts['duplicates'], lang)} duplicate rows were removed." if en else
                              f"{num(facts['duplicates'], lang)} doppelte Zeilen wurden entfernt."))
        if stats["filled"]:
            imputation = facts.get("imputation")
            how = ""
            if imputation:
                strategy = STRATEGY["imputation"][imputation["strategy"]][lang]
                how = ((f" (numbers with {strategy}{', after a quick test' if imputation.get('evaluated') else ''})") if en else
                       (f" (Zahlen mit {strategy}{', nach einem Schnelltest' if imputation.get('evaluated') else ''})"))
            sentences.append((f"{num(stats['filled'], lang)} missing values were filled{how}." if en else
                              f"{num(stats['filled'], lang)} fehlende Werte wurden ergänzt{how}."))
        if facts.get("missing_category") and facts["missing_category"]["columns"]:
            sentences.append((f"Where most values were missing ({listing(facts['missing_category']['columns'], lang)}), gaps were marked instead of guessed." if en else
                              f"Wo die meisten Werte fehlten ({listing(facts['missing_category']['columns'], lang)}), wurden Lücken markiert statt geraten."))
        if facts["identifiers"]:
            sentences.append((f"{listing(facts['identifiers'], lang)} {'were' if len(facts['identifiers']) > 1 else 'was'} kept for reference but not used for analysis." if en else
                              f"{listing(facts['identifiers'], lang)} {'bleiben' if len(facts['identifiers']) > 1 else 'bleibt'} zur Orientierung erhalten, {'werden' if len(facts['identifiers']) > 1 else 'wird'} aber nicht ausgewertet."))
        if facts.get("target"):
            task = {"classification": ("a classification", "eine Klassifikation"), "regression": ("a regression", "eine Regression")}[facts["task_type"]]
            sentences.append((f"Predicting {quote(facts['target'], lang)} was treated as {task[0]} task, and each preparation choice was tested against it." if en else
                              f"Die Vorhersage von {quote(facts['target'], lang)} wurde als {task[1]} behandelt, und jede Aufbereitungswahl wurde daran getestet."))
        quality_before, quality_after = stats["quality_before"], stats["quality_after"]
        sentences.append((f"Data quality went from {round(quality_before * 100)}% to {round(quality_after * 100)}% of cells needing no fix." if en else
                          f"Die Datenqualität stieg von {round(quality_before * 100)} % auf {round(quality_after * 100)} % Zellen ohne Korrekturbedarf."))
        return " ".join(sentences)

    return {**both(render), "kind": "automatic"}

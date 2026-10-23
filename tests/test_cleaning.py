import numpy as np
import pandas as pd
import pytest

from data_preparation import _category_mapping, _is_identifier, _parse_numeric_text


@pytest.mark.parametrize("text, expected", [
    ("$75", 75), ("USD 30", 30), ("30 EUR", 30), ("84%", 84), ("1,200", 1200), (" 12.5 ", 12.5),
])
def test_numeric_text_is_parsed(text, expected):
    assert _parse_numeric_text(pd.Series([text]))[0] == expected


def test_unparseable_text_becomes_missing():
    parsed = _parse_numeric_text(pd.Series(["abc", None, "7"]))
    assert np.isnan(parsed[0]) and np.isnan(parsed[1]) and parsed[2] == 7


def test_case_variants_collapse_to_most_frequent_spelling():
    mapping = _category_mapping(pd.Series(["North"] * 3 + ["NORTH", "north ", "South"]))
    assert mapping == {"NORTH": "North", "north ": "North"}


def test_single_letter_merges_only_when_unambiguous():
    assert _category_mapping(pd.Series(["male", "male", "m"])) == {"m": "male"}
    # 'a' could be 'apple' or 'avocado': leave it alone
    assert _category_mapping(pd.Series(["apple", "avocado", "a"])) == {}


def test_single_letter_codes_are_kept_in_high_cardinality_columns():
    # Ticker-style codes: 'M' is a symbol of its own, not short for 'MNT'
    symbols = ["M", "MNT", "BTC", "ETH", "SOL", "ADA", "DOT", "XRP", "BNB", "TRX", "LTC", "LINK"]
    assert _category_mapping(pd.Series(symbols * 3)) == {}


def test_duplicate_rows_do_not_hide_identifier_columns(run_pipeline):
    df = pd.DataFrame({
        "full_name": [f"Person {i}" for i in range(18)],
        "x": np.arange(18, dtype=float),
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact duplicate row
    result = run_pipeline(df)

    assert "full_name" not in result.processed.columns


def test_identifier_detection():
    n = 50
    assert _is_identifier("name", pd.Series([f"Person {i}" for i in range(n)]))
    assert _is_identifier("customer_id", pd.Series(range(n)))
    # Unique numbers that are not named like IDs are real measurements
    assert not _is_identifier("price", pd.Series(np.linspace(1, 2, n)))
    assert not _is_identifier("paid", pd.Series(range(n)))
    assert not _is_identifier("region", pd.Series(["north", "south"] * (n // 2)))


def test_mostly_text_column_is_not_forced_numeric(run_pipeline):
    df = pd.DataFrame({
        "code": ["A1", "B2", "7", "C3"] * 15,
        "x": np.arange(60, dtype=float),
    })
    result = run_pipeline(df)

    assert result.human_readable["code"].dtype == object


def test_unique_dates_are_parsed_not_dropped_as_identifiers(run_pipeline):
    df = pd.DataFrame({
        "signup_date": pd.date_range("2020-01-01", periods=60, freq="7D").strftime("%Y-%m-%d"),
        "x": np.arange(60, dtype=float),
    })
    result = run_pipeline(df)

    assert {"signup_date_year", "signup_date_month", "signup_date_day"} <= set(result.processed.columns)


def test_binary_and_integer_columns_stay_whole_after_imputation(run_pipeline):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "flag": rng.choice(["yes", "no"], size=80),
        "age": rng.integers(18, 70, size=80).astype(float),
        "x": rng.normal(size=80),
    })
    df.loc[:9, "flag"] = np.nan
    df.loc[10:19, "age"] = np.nan
    result = run_pipeline(df)

    assert set(result.human_readable["flag"].unique()) <= {0, 1}
    assert (result.human_readable["age"] % 1 == 0).all()


# --- German / European formats --------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("89,90 €", 89.90), ("EUR 89.90", 89.90), ("249 €", 249), ("1.299,00 €", 1299), ("12,5 %", 12.5),
    ("CHF 12.50", 12.50), ("1,200.50", 1200.50), ("-3,5", -3.5),
])
def test_european_and_english_number_formats(text, expected):
    assert _parse_numeric_text(pd.Series([text]))[0] == pytest.approx(expected)


def test_dot_thousands_need_decimal_comma_context():
    # In a column that uses decimal commas, "1.299" is one thousand two hundred ninety-nine...
    assert list(_parse_numeric_text(pd.Series(["1.299 €", "89,90 €"]))) == pytest.approx([1299, 89.90])
    # ...otherwise it stays an ordinary decimal number
    assert _parse_numeric_text(pd.Series(["1.299"]))[0] == pytest.approx(1.299)


def test_german_yes_no_becomes_zero_one(run_pipeline):
    df = pd.DataFrame({"im_sale": ["Ja", "nein", "JA", "Nein"] * 15, "x": np.arange(60, dtype=float)})
    result = run_pipeline(df)

    assert list(result.human_readable["im_sale"][:4]) == [1, 0, 1, 0]


def test_dotted_dates_are_day_first(run_pipeline):
    dates = pd.date_range("2020-01-01", periods=60, freq="5D")
    df = pd.DataFrame({"datum": dates.strftime("%d.%m.%Y"), "x": np.arange(60, dtype=float)})
    result = run_pipeline(df)

    assert list(result.human_readable["datum_month"]) == list(dates.month)
    assert list(result.human_readable["datum_day"]) == list(dates.day)


# --- Categories and identifiers ---------------------------------------------------

def test_merged_abbreviations_take_the_most_frequent_spelling():
    sizes = pd.Series(["M"] * 5 + ["medium"] * 2 + ["m"] + ["L"] * 3 + ["large"])
    assert _category_mapping(sizes) == {"medium": "M", "m": "M", "large": "L"}


def test_mostly_missing_category_gets_its_own_label(run_pipeline):
    rng = np.random.default_rng(6)
    cabin = rng.choice(["B5", "C22", "A23", "D10"], size=80).astype(object)
    cabin[:60] = None  # 75% missing
    port = rng.choice(["S", "C", "Q"], size=80, p=[0.7, 0.2, 0.1]).astype(object)
    port[:3] = None  # under 10% missing
    df = pd.DataFrame({"cabin": cabin, "port": port, "x": rng.normal(size=80)})
    result = run_pipeline(df)

    hr = result.human_readable
    assert (hr["cabin"] == "(missing)").sum() == 60
    assert "(missing)" not in set(hr["port"])
    assert hr["port"].notna().all()


def test_repeating_id_columns_are_identifiers():
    customers = pd.Series(np.repeat(np.arange(17000, 17100), 3).astype(float))  # 100 customers, 3 orders each
    assert _is_identifier("CustomerID", customers)
    assert _is_identifier("InvoiceNo", pd.Series([f"5363{i:02d}" for i in range(50)] * 2))
    # A small set of repeating codes is a real category, and 'number' alone is not an ID name
    assert not _is_identifier("store_id", pd.Series([1, 2, 3, 4, 5] * 20))
    assert not _is_identifier("number_of_rooms", pd.Series(range(100)))


def test_missing_identifier_values_are_kept_empty_not_invented(run_pipeline):
    rng = np.random.default_rng(7)
    customer = np.repeat(np.arange(17000, 17040), 3).astype(float)
    customer[[5, 50, 90]] = np.nan
    df = pd.DataFrame({"CustomerID": customer, "qty": rng.integers(1, 20, size=120), "x": rng.normal(size=120)})
    result = run_pipeline(df)

    assert "CustomerID" not in result.processed.columns
    assert result.human_readable["CustomerID"].isna().sum() == 3


def test_messy_numeric_target_is_repaired_and_treated_as_regression(run_pipeline):
    rng = np.random.default_rng(8)
    x = rng.normal(size=120)
    prices = 50 + 10 * x
    formats = rng.choice(["{:.2f} €", "EUR {:.2f}"], size=120)
    df = pd.DataFrame({"x": x, "price": [f.format(p).replace(".", ",") if "€" in f else f.format(p)
                                         for f, p in zip(formats, prices)]})
    result = run_pipeline(df, "price")

    assert result.decisions("Task Detection")[0]["decision"] == "Regression"
    np.testing.assert_allclose(np.sort(result.processed["price"]), np.sort(np.round(prices, 2)), atol=1e-9)
    assert result.winner("Scaling")  # candidates were scored against the repaired target


def test_id_columns_are_kept_exactly_as_written(run_pipeline):
    # Mostly numeric invoice numbers with a few cancellations ("C...") must not be number-parsed
    invoices = [str(536365 + i) for i in range(95)] + [f"C{536500 + i}" for i in range(5)]
    df = pd.DataFrame({"InvoiceNo": invoices, "qty": np.arange(100), "x": np.linspace(0, 1, 100)})
    result = run_pipeline(df)

    assert not [d for d in result.decisions("Standardization") if d["details"]["column"] == "InvoiceNo"]
    assert result.human_readable["InvoiceNo"].astype(str).str.startswith("C").sum() == 5


def test_capped_outliers_keep_real_values_in_the_readable_file(run_pipeline):
    rng = np.random.default_rng(10)
    quantity = rng.integers(1, 30, size=200).astype(float)
    quantity[7] = 1400.0  # a real bulk order, rare enough to be capped for models
    df = pd.DataFrame({"quantity": quantity, "x": rng.normal(size=200)})
    result = run_pipeline(df)

    assert result.human_readable["quantity"].max() == 1400
    assert any(d["decision"] == "Capping Applied" for d in result.decisions("Outlier Handling"))
    # ...while the ML-ready (scaled) column was capped before scaling
    ml = result.processed["quantity"]
    assert ml.max() - ml.median() < 10 * (ml.quantile(0.75) - ml.quantile(0.25))


def test_high_cardinality_text_gaps_are_not_filled_with_a_guess(run_pipeline):
    rng = np.random.default_rng(11)
    descriptions = np.array([f"PRODUCT {i}" for i in rng.integers(0, 300, size=400)], dtype=object)
    descriptions[[3, 50]] = None  # under 10% missing, but many distinct values
    df = pd.DataFrame({"description": descriptions, "qty": rng.integers(1, 9, size=400)})
    result = run_pipeline(df)

    assert (result.human_readable["description"] == "(missing)").sum() == 2

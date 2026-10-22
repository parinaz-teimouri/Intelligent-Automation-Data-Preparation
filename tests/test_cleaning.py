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

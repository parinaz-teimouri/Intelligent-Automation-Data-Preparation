import numpy as np
import pandas as pd

from data_preparation import AutoDataPreparer, read_csv_file


def test_german_excel_export_semicolons_and_cp1252(tmp_path):
    path = tmp_path / "export.csv"
    path.write_bytes("Artikel;Größe;Preis\nPullover;M;89,90 €\nMütze;Einheitsgröße;19,00 €\n".encode("cp1252"))

    df, file_format = read_csv_file(str(path))

    assert file_format == {"encoding": "cp1252", "delimiter": ";"}
    assert list(df.columns) == ["Artikel", "Größe", "Preis"]
    assert df.loc[1, "Artikel"] == "Mütze"


def test_quoted_decimal_commas_do_not_fool_the_delimiter(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text('sku,price,stock\nKV-1,"89,90 €",3\nKV-2,"1.299,00 €",1\n', encoding="utf-8")

    df, file_format = read_csv_file(str(path))

    assert file_format["delimiter"] == ","
    assert df.shape == (2, 3)


def test_tab_separated_with_utf8_bom(tmp_path):
    path = tmp_path / "data.tsv.csv"
    path.write_bytes("﻿a\tb\n1\t2\n3\t4\n".encode("utf-8"))

    df, file_format = read_csv_file(str(path))

    assert file_format == {"encoding": "utf-8-sig", "delimiter": "\t"}
    assert list(df.columns) == ["a", "b"]


def test_german_export_is_cleaned_end_to_end(tmp_path):
    rng = np.random.default_rng(9)
    lines = ["Artikel;Farbe;Preis;Im Sale"]
    for i in range(60):
        colour = rng.choice(["Navy", "NAVY", "navy "])
        lines.append(f"Pullover {i};{colour};{rng.integers(20, 200)},{rng.integers(0, 99):02d} €;{rng.choice(['Ja', 'Nein'])}")
    path = tmp_path / "export.csv"
    path.write_bytes(("\n".join(lines) + "\n").encode("cp1252"))

    preparer = AutoDataPreparer(str(path), models_dir=str(tmp_path / "models"), record_feedback=False)
    preparer.run_pipeline(str(tmp_path / "processed_export.csv"))
    clean = preparer.human_readable_data

    assert pd.api.types.is_numeric_dtype(clean["Preis"])
    assert set(clean["Farbe"]) == {"Navy"}
    assert set(clean["Im Sale"]) <= {0, 1}


def test_step_timings_and_clean_table_are_kept(run_pipeline, tmp_path):
    from conftest import make_classification_df
    df = make_classification_df()
    input_path = tmp_path / "in.csv"
    df.to_csv(input_path, index=False)
    preparer = AutoDataPreparer(str(input_path), target_col="segment", models_dir=str(tmp_path / "m"), record_feedback=False)
    preparer.run_pipeline(str(tmp_path / "processed_in.csv"))

    assert set(preparer.step_seconds) == {"load", "clean", "duplicates", "impute", "encode", "scale", "report"}
    assert all(seconds >= 0 for seconds in preparer.step_seconds.values())
    saved = pd.read_csv(tmp_path / "human_readable_in.csv")
    assert preparer.human_readable_data.shape == saved.shape
    # Row labels still point at the raw rows, so cells can be compared before/after
    assert set(preparer.human_readable_data.index) <= set(preparer.raw_data_snapshot.index)

import io

import pandas as pd

from api.shopify import COLUMNS, map_columns, slug, to_shopify_csv


def catalog():
    return pd.DataFrame({
        "Artikel": ["Merino Crew Knit", "Merino Crew Knit", "Merino Crew Knit", "Silk Scarf"],
        "Farbe": ["Navy", "Navy", "Black", "Sand"],
        "Größe": ["S", "M", "S", "One Size"],
        "Preis": [89.9, 89.9, 89.9, 49.0],
        "Bestand": [12, 0, 5, 3],
        "SKU": ["KV-1", "KV-2", "KV-3", "KV-4"],
        "Kategorie": ["Knitwear", "Knitwear", "Knitwear", "Accessories"],
        "material": ["Merino Wool", "Merino Wool", "Merino Wool", "Silk"],
        "on_sale": [1, 1, 1, 0],
    })


def read(csv_text):
    return pd.read_csv(io.StringIO(csv_text), keep_default_na=False, dtype=str)


def test_german_column_names_are_recognised():
    roles = map_columns(catalog())
    assert roles["title"] == "Artikel" and roles["price"] == "Preis" and roles["size"] == "Größe"
    assert roles["stock"] == "Bestand" and roles["type"] == "Kategorie"


def test_variants_share_a_handle_and_only_the_first_row_has_product_fields():
    shop = read(to_shopify_csv(catalog(), vendor="Atelier Kova"))

    assert list(shop.columns) == COLUMNS
    knit = shop[shop["URL handle"] == "merino-crew-knit"]
    assert len(knit) == 3
    assert knit["Title"].tolist() == ["Merino Crew Knit", "", ""]
    assert knit["Vendor"].tolist() == ["Atelier Kova", "", ""]
    assert set(knit["Option1 name"]) == {"Colour"} and set(knit["Option2 name"]) == {"Size"}
    assert knit["Price"].tolist() == ["89.90"] * 3
    assert knit.iloc[0]["Tags"] == "Merino Wool, sale"
    assert set(shop["Status"]) == {"draft", ""}


def test_repeated_variants_across_seasons_become_separate_products():
    df = pd.concat([catalog().assign(season="SS25"), catalog().assign(season="AW25")], ignore_index=True)
    shop = read(to_shopify_csv(df))

    handles = set(shop["URL handle"])
    assert {"merino-crew-knit-ss25", "merino-crew-knit-aw25"} <= handles
    assert not shop.duplicated(["URL handle", "Option1 value", "Option2 value"]).any()


def test_non_catalogs_get_no_shopify_file():
    passengers = pd.DataFrame({"name": ["Allen, Miss. Elisabeth"], "age": [29], "fare": [211.34]})
    assert to_shopify_csv(passengers) is None


def test_slug_handles_umlauts():
    assert slug("Größe Übergangsjacke") == "groesse-uebergangsjacke"

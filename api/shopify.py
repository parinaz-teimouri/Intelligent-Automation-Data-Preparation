"""
Exports a cleaned product catalog in Shopify's product CSV format
(https://help.shopify.com/en/manual/products/import-export/using-csv).

Columns are recognised by common English and German names. Rows sharing a URL handle are variants
of one product; the first row carries the product fields and later rows only the variant fields.
Products are imported as drafts, so nothing goes live before someone has looked at it.
"""
import re
import unicodedata

import pandas as pd

SYNONYMS = {
    "title": ["product", "title", "name", "product_name", "productname", "artikel", "artikelname",
              "bezeichnung", "produkt", "produktname"],
    "price": ["price", "preis", "unit_price", "unitprice", "sale_price", "verkaufspreis", "vk", "vk_preis"],
    "sku": ["sku", "article_number", "artikelnummer", "item_number", "art_nr", "artnr"],
    "colour": ["colour", "color", "farbe"],
    "size": ["size", "größe", "groesse", "grosse"],
    "stock": ["stock", "inventory", "qty", "quantity", "bestand", "lagerbestand", "lager"],
    "type": ["category", "type", "product_type", "kategorie", "warengruppe", "produkttyp"],
    "vendor": ["brand", "vendor", "marke", "hersteller"],
    "description": ["description", "beschreibung", "body"],
    "on_sale": ["on_sale", "sale", "im_sale", "reduziert"],
    "tags": ["material", "season", "saison", "collection", "kollektion"],
}
COLUMNS = ["Title", "URL handle", "Description", "Vendor", "Type", "Tags", "Published on online store", "Status",
           "SKU", "Option1 name", "Option1 value", "Option2 name", "Option2 value", "Price",
           "Inventory tracker", "Inventory quantity", "Continue selling when out of stock",
           "Requires shipping", "Charge tax"]
PRODUCT_ONLY = ["Title", "Description", "Vendor", "Type", "Tags", "Published on online store", "Status"]


def _normalise(name):
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def map_columns(df):
    """{role: column} for the columns that could be recognised."""
    by_name = {_normalise(c): c for c in df.columns}
    mapping = {}
    for role, names in SYNONYMS.items():
        found = [by_name[n] for n in names if n in by_name]
        if found:
            mapping[role] = found if role == "tags" else found[0]
    return mapping


def slug(text):
    text = str(text).lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _text(value):
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()


def to_shopify_csv(df, vendor=None):
    """The Shopify product CSV as a string, or None when the table does not look like a product catalog."""
    roles = map_columns(df)
    if "title" not in roles or "price" not in roles or not pd.api.types.is_numeric_dtype(df[roles["price"]]):
        return None

    options = [(role, label) for role, label in [("colour", "Colour"), ("size", "Size")] if role in roles]
    option_columns = [roles[role] for role, _ in options]
    handles = df[roles["title"]].map(slug)
    # The same product in several seasons/collections becomes separate products if its variants repeat
    if option_columns and "tags" in roles:
        repeated = df.assign(_h=handles).duplicated(["_h"] + option_columns, keep=False)
        if repeated.any():
            split_by = roles["tags"][-1]
            handles = handles.where(~repeated, handles + "-" + df[split_by].map(slug))
    handles = handles + df.assign(_h=handles).groupby(["_h"] + option_columns, dropna=False).cumcount().map(
        lambda n: "" if n == 0 else f"-{n + 1}")

    rows = []
    for (handle, group) in df.assign(_handle=handles).groupby("_handle", sort=False):
        for position, (_, item) in enumerate(group.iterrows()):
            tags = [_text(item[c]) for c in roles.get("tags", []) if _text(item[c])]
            if "on_sale" in roles and _text(item[roles["on_sale"]]) in {"1", "1.0", "True", "true"}:
                tags.append("sale")
            row = {
                "Title": _text(item[roles["title"]]),
                "URL handle": handle,
                "Description": _text(item[roles["description"]]) if "description" in roles else "",
                "Vendor": _text(item[roles["vendor"]]) if "vendor" in roles else (vendor or ""),
                "Type": _text(item[roles["type"]]) if "type" in roles else "",
                "Tags": ", ".join(tags),
                "Published on online store": "true",
                "Status": "draft",
                "SKU": _text(item[roles["sku"]]) if "sku" in roles else "",
                "Price": "" if pd.isna(item[roles["price"]]) else f"{float(item[roles['price']]):.2f}",
                "Inventory tracker": "shopify" if "stock" in roles else "",
                "Inventory quantity": "" if "stock" not in roles or pd.isna(item[roles["stock"]]) else str(int(item[roles["stock"]])),
                "Continue selling when out of stock": "deny",
                "Requires shipping": "true",
                "Charge tax": "true",
            }
            if options:
                for number, (role, label) in enumerate(options, start=1):
                    row[f"Option{number} name"] = label
                    row[f"Option{number} value"] = _text(item[roles[role]])
            else:
                row["Option1 name"], row["Option1 value"] = "Title", "Default Title"
            if position > 0:
                for column in PRODUCT_ONLY:
                    row[column] = ""
            rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS).fillna("").to_csv(index=False)

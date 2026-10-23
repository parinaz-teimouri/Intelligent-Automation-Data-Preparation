"""
Builds the sample datasets used by the demo app (see SOURCES.md for credits and licences).

  fashion_catalog.csv       generated: product catalog of the fictional brand "Atelier Kova",
                            with the kinds of defects found in real shop exports
  online_retail_sample.csv  UCI Online Retail (CC BY 4.0): every line of randomly chosen invoices
  titanic.csv               OpenML 40945 (public): 'boat' and 'body' removed, they reveal the outcome
  penguins.csv              palmerpenguins (CC0)

Downloads are cached in ~/.cache/data-prep-samples. Usage: python samples/build_samples.py
"""
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SAMPLES_DIR)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "data-prep-samples")

ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
PENGUINS_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"


def _use_certifi():
    # python.org builds on macOS ship without CA certificates
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def _download(url, filename):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        _use_certifi()
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url) as response, open(path, "wb") as f:
            f.write(response.read())
    return path


# --- Fashion catalog (generated) -----------------------------------------------------

PRODUCTS = [  # category, product, base price (EUR), material
    ("Knitwear", "Merino Crew Knit", 89.90, "Merino Wool"),
    ("Knitwear", "Cashmere Cardigan", 229.00, "Cashmere"),
    ("Knitwear", "Cable Knit Jumper", 119.00, "Wool"),
    ("Knitwear", "Ribbed Turtleneck", 99.00, "Merino Wool"),
    ("Shirts", "Linen Shirt", 59.00, "Linen"),
    ("Shirts", "Oxford Shirt", 69.00, "Cotton"),
    ("Shirts", "Silk Blouse", 139.00, "Silk"),
    ("Shirts", "Poplin Shirt", 65.00, "Cotton"),
    ("Outerwear", "Wool Coat", 249.00, "Wool"),
    ("Outerwear", "Cashmere Coat", 1299.00, "Cashmere"),
    ("Outerwear", "Quilted Jacket", 179.00, "Polyester"),
    ("Outerwear", "Trench Coat", 299.00, "Cotton"),
    ("Trousers", "Pleated Trousers", 99.00, "Wool"),
    ("Trousers", "Wide-Leg Chinos", 79.00, "Cotton"),
    ("Trousers", "Linen Trousers", 85.00, "Linen"),
    ("Dresses", "Midi Dress", 129.00, "Viscose"),
    ("Dresses", "Slip Dress", 109.00, "Silk"),
    ("Dresses", "Knit Dress", 149.00, "Merino Wool"),
    ("Accessories", "Silk Scarf", 49.00, "Silk"),
    ("Accessories", "Leather Belt", 39.00, "Leather"),
]
COLOURS = ["Navy", "Black", "Camel", "Off White", "Forest Green", "Burgundy", "Grey Melange", "Sand"]
SIZES = ["XS", "S", "M", "L", "XL"]
SIZE_VARIANTS = {"S": ["s", "small", "Small"], "M": ["m", "medium", "Medium"], "L": ["l", "large", "Large"],
                 "XL": ["xl"], "XS": ["xs"], "One Size": ["ONE SIZE", "one size"]}
SEASONS = {"SS25": "2025-02-15", "AW25": "2025-08-15"}
TRUE_SPELLINGS = ["Ja", "ja", "1", "yes", "TRUE"]
FALSE_SPELLINGS = ["nein", "Nein", "0", "No", "false"]


def _format_price(value, rng):
    """Prices exported by different systems: German, English and plain notation."""
    style = rng.choice(["de", "en_code", "plain", "en_symbol"], p=[0.5, 0.25, 0.15, 0.10])
    if style == "de":
        whole, cents = f"{value:,.2f}".split(".")
        return f"{whole.replace(',', '.')},{cents} €"
    if style == "en_code":
        return f"EUR {value:,.2f}"
    if style == "en_symbol":
        return f"€{value:.2f}"
    return f"{value:.2f}"


def _messy(value, variants, rate, rng):
    return rng.choice(variants) if variants and rng.random() < rate else value


def build_fashion_catalog(n_rows=1200, n_duplicates=6, seed=2025):
    rng = np.random.default_rng(seed)
    rows = []
    for season, launch in SEASONS.items():
        for category, product, base_price, material in PRODUCTS:
            colours = rng.choice(COLOURS, size=7, replace=False)
            sizes = ["One Size"] if category == "Accessories" else SIZES
            for colour in colours:
                for size in sizes:
                    rows.append((season, launch, category, product, base_price, material, colour, size))
    picked = rng.choice(len(rows), size=n_rows - n_duplicates, replace=False)
    picked.sort()

    records = []
    for i, idx in enumerate(picked):
        season, launch, category, product, base_price, material, colour, size = rows[idx]
        launch_date = pd.Timestamp(launch) + pd.Timedelta(days=int(rng.integers(-20, 40)))
        stock = int(rng.poisson(14))
        weeks_live = (pd.Timestamp("2025-11-01") - launch_date).days / 7
        # Older lines with plenty of stock tend to go on sale: a learnable target
        p_sale = 1 / (1 + np.exp(-(-2.2 + 0.08 * (stock - 14) + 0.06 * weeks_live)))
        on_sale = rng.random() < p_sale
        rating = np.clip(rng.normal(4.2, 0.45), 1, 5) if rng.random() > 0.12 else None
        records.append({
            "sku": f"KV-{10001 + i}",
            "product": product,
            "category": category,
            "colour": _messy(colour, [colour.upper(), colour.lower(), colour + " "], 0.25, rng),
            "size": _messy(size, SIZE_VARIANTS.get(size), 0.2, rng),
            "material": _messy(material, [material.upper(), material.lower()], 0.15, rng),
            "season": season,
            "price": _format_price(base_price, rng),
            "stock": stock if rng.random() > 0.035 else None,
            "on_sale": rng.choice(TRUE_SPELLINGS if on_sale else FALSE_SPELLINGS),
            "launch_date": launch_date.strftime("%d.%m.%Y"),
            "rating": None if rating is None else f"{rating:.1f}".replace(".", ","),
        })
    df = pd.DataFrame(records)
    df["stock"] = df["stock"].astype("Int64")
    duplicates = df.sample(n_duplicates, random_state=seed)
    df = pd.concat([df, duplicates]).sort_values("sku", kind="stable").reset_index(drop=True)
    return df


# --- Real datasets -------------------------------------------------------------------

def build_online_retail(min_rows=3000, seed=0):
    with zipfile.ZipFile(_download(ONLINE_RETAIL_URL, "online_retail.zip")) as archive:
        xlsx = next(name for name in archive.namelist() if name.endswith(".xlsx"))
        print("Reading Online Retail.xlsx (takes a minute) ...")
        full = pd.read_excel(io.BytesIO(archive.read(xlsx)))
    # Whole invoices keep duplicate lines and repeat customers as they occur in the source
    rng = np.random.default_rng(seed)
    invoices = rng.permutation(full["InvoiceNo"].astype(str).unique())
    lines_per_invoice = full["InvoiceNo"].astype(str).value_counts()
    chosen = invoices[: int(np.searchsorted(lines_per_invoice[invoices].cumsum().to_numpy(), min_rows)) + 1]
    sample = full[full["InvoiceNo"].astype(str).isin(chosen)].copy()
    sample["CustomerID"] = sample["CustomerID"].astype("Int64")
    return sample.reset_index(drop=True)


def build_titanic():
    _use_certifi()
    from sklearn.datasets import fetch_openml
    frame = fetch_openml("titanic", version=1, as_frame=True, data_home=CACHE_DIR).frame
    return frame.drop(columns=["boat", "body"])


def build_penguins():
    return pd.read_csv(_download(PENGUINS_URL, "penguins.csv"))


CATALOG = [
    {"id": "fashion", "file": "samples/fashion_catalog.csv", "shopify": True,
     "targets": ["on_sale", "price", "rating"],
     "name": {"en": "Fashion catalog", "de": "Mode-Katalog"},
     "domain": {"en": "E-commerce", "de": "E-Commerce"},
     "source": {"en": "Atelier Kova, a fictional brand. Generated by samples/build_samples.py.",
                "de": "Atelier Kova, eine fiktive Marke. Erzeugt von samples/build_samples.py."},
     "licence": "CC0 (generated)", "url": None},
    {"id": "retail", "file": "samples/online_retail_sample.csv", "shopify": False,
     "targets": ["Country", "Quantity"],
     "name": {"en": "Online shop orders", "de": "Online-Shop-Bestellungen"},
     "domain": {"en": "E-commerce", "de": "E-Commerce"},
     "source": {"en": "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. All lines of randomly chosen invoices.",
                "de": "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. Alle Zeilen zufällig gewählter Rechnungen."},
     "licence": "CC BY 4.0", "url": "https://doi.org/10.24432/C5BW33"},
    {"id": "titanic", "file": "samples/titanic.csv", "shopify": False,
     "targets": ["survived"],
     "name": {"en": "Titanic passengers", "de": "Titanic-Passagiere"},
     "domain": {"en": "History", "de": "Geschichte"},
     "source": {"en": "Harrell & Cason, Vanderbilt Biostatistics, via OpenML. 'boat' and 'body' removed: they reveal the outcome.",
                "de": "Harrell & Cason, Vanderbilt Biostatistics, über OpenML. 'boat' und 'body' entfernt: Sie verraten das Ergebnis."},
     "licence": "Public (OpenML)", "url": "https://www.openml.org/d/40945"},
    {"id": "freelancers", "file": "input/global_freelancers_raw.csv", "shopify": False,
     "targets": ["rating"],
     "name": {"en": "Freelancer profiles", "de": "Freelancer-Profile"},
     "domain": {"en": "HR", "de": "HR"},
     "source": {"en": "Urvish Ahir, Global Freelancers (Raw) Dataset, Kaggle. Fictional profiles generated with Faker.",
                "de": "Urvish Ahir, Global Freelancers (Raw) Dataset, Kaggle. Fiktive, mit Faker erzeugte Profile."},
     "licence": "CC0", "url": "https://www.kaggle.com/datasets/urvishahir/global-freelancers-raw-dataset"},
    {"id": "penguins", "file": "samples/penguins.csv", "shopify": False,
     "targets": ["species", "body_mass_g", "sex"],
     "name": {"en": "Penguin measurements", "de": "Pinguin-Messdaten"},
     "domain": {"en": "Science", "de": "Wissenschaft"},
     "source": {"en": "Horst, Hill & Gorman (2020). palmerpenguins. Palmer Station LTER.",
                "de": "Horst, Hill & Gorman (2020). palmerpenguins. Palmer Station LTER."},
     "licence": "CC0", "url": "https://doi.org/10.5281/zenodo.3960218"},
]


if __name__ == "__main__":
    builders = {
        "samples/fashion_catalog.csv": build_fashion_catalog,
        "samples/online_retail_sample.csv": build_online_retail,
        "samples/titanic.csv": build_titanic,
        "samples/penguins.csv": build_penguins,
    }
    for relative_path, build in builders.items():
        df = build()
        df.to_csv(os.path.join(REPO_DIR, relative_path), index=False)
        print(f"Wrote {relative_path}: {len(df)} rows x {df.shape[1]} columns")

    catalog = []
    for entry in CATALOG:
        df = pd.read_csv(os.path.join(REPO_DIR, entry["file"]))
        catalog.append({**entry, "rows": len(df), "columns": df.shape[1]})
    with open(os.path.join(SAMPLES_DIR, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print("Wrote samples/catalog.json")

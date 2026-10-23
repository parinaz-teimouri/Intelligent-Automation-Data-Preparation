# Sample datasets: sources and licences

The files in this folder are rebuilt by `python samples/build_samples.py`;
`catalog.json` describes them for the demo app.

| File | Source | Licence | Changes |
|---|---|---|---|
| `fashion_catalog.csv` | Generated for this project. "Atelier Kova" is a fictional brand; products, prices and stock are invented. | CC0 | Defects added on purpose: spelling variants, German and English price formats, yes/no text in several languages, missing values, duplicate rows. |
| `online_retail_sample.csv` | Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | All lines of randomly chosen invoices (seed 0), about 3,000 of the original 541,909 rows. Dates written as ISO timestamps. |
| `titanic.csv` | Frank E. Harrell Jr. and Thomas Cason, Vanderbilt Biostatistics, via [OpenML dataset 40945](https://www.openml.org/d/40945) | Public (as listed on OpenML) | Columns `boat` and `body` removed: they were recorded after the sinking and reveal the outcome. |
| `../input/global_freelancers_raw.csv` | Urvish Ahir, *Global Freelancers (Raw) Dataset*, Kaggle. https://www.kaggle.com/datasets/urvishahir/global-freelancers-raw-dataset. 1,000 fictional profiles generated with the Faker library. | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | None. |
| `penguins.csv` | Horst, A. M., Hill, A. P. and Gorman, K. B. (2020). *palmerpenguins*. https://doi.org/10.5281/zenodo.3960218. Data collected by Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research Program. | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | None. |

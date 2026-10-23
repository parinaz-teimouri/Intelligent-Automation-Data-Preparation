# Benchmark results

Generated 2026-10-23 08:18 by `benchmark.py` (5 repeats, scikit-learn 1.8.0, pandas 2.3.3). Meta-models: none trained, fixed rules used (run train_meta_model.py).

Score = macro F1 (classification) or R² (regression) on the evaluation half, 5-fold CV, averaged over a linear model and a random forest. Mean (std over repeats); best per row in bold.

| Dataset | Task | static | static + cleaning | meta-model only | full system |
| --- | --- | ---: | ---: | ---: | ---: |
| adult (OpenML) | classification | **0.772 (0.015)** | **0.772 (0.015)** | 0.772 (0.015) | 0.772 (0.015) |
| breast_cancer (defects) | classification | 0.936 (0.010) | 0.950 (0.010) | 0.949 (0.007) | **0.953 (0.011)** |
| credit-g (OpenML) | classification | 0.661 (0.013) | 0.667 (0.015) | 0.668 (0.015) | **0.668 (0.015)** |
| diabetes (defects) | regression | 0.280 (0.054) | 0.304 (0.027) | **0.309 (0.036)** | 0.306 (0.030) |
| freelancers (repo) | regression | -0.155 (0.042) | -0.111 (0.023) | **-0.111 (0.027)** | -0.112 (0.024) |
| titanic (OpenML) | classification | 0.777 (0.019) | **0.781 (0.020)** | 0.779 (0.020) | 0.777 (0.017) |
| wine (defects) | classification | 0.941 (0.024) | 0.961 (0.024) | 0.964 (0.015) | **0.971 (0.014)** |
| **Mean rank** (1 = best) |  | 3.50 | 2.21 | 2.07 | 2.21 |
| Mean preparation time (s) |  | 0.17 | 0.12 | 0.08 | 0.36 |

## Head to head

Each dataset × repeat is one paired comparison on the identical split (score averaged over both models).

| Comparison | Wins | Ties | Losses | Mean difference |
| --- | ---: | ---: | ---: | ---: |
| static + cleaning vs static | 24 | 5 | 6 | +0.0159 |
| full system vs static + cleaning | 18 | 5 | 12 | +0.0016 |
| full system vs meta-model only | 10 | 17 | 8 | +0.0006 |
| full system vs static | 28 | 0 | 7 | +0.0175 |

## Per downstream model

| Dataset | Model | static | static + cleaning | meta-model only | full system |
| --- | --- | ---: | ---: | ---: | ---: |
| breast_cancer (defects) | linear | 0.946 | 0.958 | 0.952 | 0.963 |
| breast_cancer (defects) | random forest | 0.925 | 0.943 | 0.945 | 0.943 |
| wine (defects) | linear | 0.931 | 0.963 | 0.954 | 0.968 |
| wine (defects) | random forest | 0.951 | 0.960 | 0.975 | 0.973 |
| diabetes (defects) | linear | 0.253 | 0.297 | 0.305 | 0.299 |
| diabetes (defects) | random forest | 0.306 | 0.310 | 0.314 | 0.313 |
| freelancers (repo) | linear | -0.214 | -0.120 | -0.121 | -0.121 |
| freelancers (repo) | random forest | -0.096 | -0.103 | -0.100 | -0.103 |
| titanic (OpenML) | linear | 0.781 | 0.781 | 0.774 | 0.771 |
| titanic (OpenML) | random forest | 0.773 | 0.781 | 0.785 | 0.783 |
| adult (OpenML) | linear | 0.770 | 0.770 | 0.771 | 0.771 |
| adult (OpenML) | random forest | 0.773 | 0.773 | 0.772 | 0.772 |
| credit-g (OpenML) | linear | 0.675 | 0.675 | 0.676 | 0.677 |
| credit-g (OpenML) | random forest | 0.647 | 0.659 | 0.659 | 0.659 |

## Number of features after preparation

| Dataset | static | static + cleaning | meta-model only | full system |
| --- | ---: | ---: | ---: | ---: |
| breast_cancer (defects) | 373 | 31 | 31 | 31 |
| wine (defects) | 198 | 14 | 14 | 14 |
| diabetes (defects) | 318 | 11 | 11 | 11 |
| freelancers (repo) | 1047 | 50 | 50 | 50 |
| titanic (OpenML) | 1543 | 888 | 11 | 11 |
| adult (OpenML) | 90 | 90 | 90 | 90 |
| credit-g (OpenML) | 48 | 48 | 48 | 48 |

## Datasets

- **breast_cancer (defects)**: 569 rows, target `diagnosis` (classification); scikit-learn breast_cancer with injected defects
- **wine (defects)**: 178 rows, target `cultivar` (classification); scikit-learn wine with injected defects
- **diabetes (defects)**: 442 rows, target `progression` (regression); scikit-learn diabetes with injected defects
- **freelancers (repo)**: 899 rows, target `rating` (regression); input/global_freelancers_raw.csv, messy real-world style export
- **titanic (OpenML)**: 1309 rows, target `survived` (classification); OpenML 'titanic' v1, real data
- **adult (OpenML)**: 3000 rows, target `class` (classification); OpenML 'adult' v2, real data
- **credit-g (OpenML)**: 1000 rows, target `class` (classification); OpenML 'credit-g' v1, real data

## Notes

- The full system chooses imputation and scaling on the selection half and is scored on the disjoint evaluation half, so its choices are not tuned on the rows it is scored on.
- Within the evaluation half, preprocessing is fitted on all its rows before cross-validation. This is identical for every configuration, so comparisons are fair, but absolute scores are slightly optimistic.
- Defects in the scikit-learn datasets are injected synthetically (`inject_defects`); the OpenML and repository datasets are used as they are.

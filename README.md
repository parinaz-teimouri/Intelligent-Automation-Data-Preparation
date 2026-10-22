# Intelligent Data Preparation Pipeline

**An automated data preprocessing framework using meta-learning and performance-driven optimization.**

This project intelligently automates data preparation decisions—imputation strategies, scaling methods, and encoding approaches—by learning which strategies perform best for different data characteristics.

## Features

- 🤖 **Meta-Learning**: Uses trained models to predict optimal preprocessing strategies
- 📊 **Auto Task Detection**: Automatically identifies classification vs. regression tasks
- 📈 **Performance-Driven**: Evaluates candidate strategies using downstream model performance
- 📝 **Detailed Reports**: Generates JSON reports tracking all preprocessing decisions
- 🎨 **Interactive Dashboard**: Streamlit-based UI for visual exploration
- 📦 **End-to-End Pipeline**: Complete automation from raw data to ML-ready dataset

## Project Structure

```
├── data_preparation.py       # Main AutoDataPreparer implementation
├── meta_learning.py          # MetaLearner & MetaFeatureExtractor classes
├── train_meta_model.py       # Generates meta-training data and trains the meta-models
├── test_pipeline.py          # Command-line runner for one data file
├── dashboard.py              # Streamlit interactive dashboard
├── tests/                    # Automated tests (pytest)
├── input/                    # Raw input datasets
├── meta_models/              # Trained meta-models and evaluation feedback
├── paper/                    # Published paper (LaTeX)
└── new-paper/                # Working paper draft (LaTeX)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Intelligent-Automation-Data-Preparation
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Run the pipeline on a CSV file:
```bash
python test_pipeline.py path/to/your/data.csv --target target_column
```
`--target` is optional: with it, candidate strategies are scored against the target
(optimisation mode); without it, strategies are only predicted. Outputs (`processed_*.csv`,
`human_readable_*.csv`, `report_*.json`) go to the current directory or `--output-dir`.

### Launch the interactive dashboard:
```bash
streamlit run dashboard.py
```

### Train meta-learning models:
```bash
python train_meta_model.py              # options: --datasets 40 --seed 0 --models-dir meta_models
```
Without trained models the pipeline falls back to fixed rules for its first guess and still
evaluates the next-best candidate.

### Run the tests:
```bash
pytest
```

### Run the benchmark:
```bash
python benchmark.py                     # options: --repeats 3 --no-openml --models-dir --output-dir
```
Compares a static pipeline, static + cleaning, meta-model prediction only, and the full system
on 7 datasets (3 OpenML datasets are downloaded on first use). Results are written to
[`benchmark_results/RESULTS.md`](benchmark_results/RESULTS.md); the protocol is described in
the docstring of `benchmark.py`.

## How It Works

1. **Target Separation**: The target column is set aside so no preprocessing step can alter it or leak it into evaluation; rows without a target are dropped
2. **Cleaning**: Text values are standardised (yes/no → 0/1, `"$75"` / `"84%"` → numbers, spelling variants merged), dates are expanded into year/month/day, duplicates are removed, and ID / free-text columns are excluded from the ML features
3. **Data Analysis**: Extracts meta-features (missing ratio, skewness, cardinality, etc.)
4. **Strategy Prediction**: Meta-models propose the top-K candidate strategies for each preprocessing task
5. **Performance Evaluation**: Each candidate is cross-validated with a downstream model (F1 for classification, R² for regression)
6. **Selection**: Chooses the strategy with the highest score and stores the result as feedback for future meta-training
7. **Reporting**: Logs all decisions, scores and failures with detailed change tracking

## Key Components

- **AutoDataPreparer**: Main orchestrator for the entire pipeline
- **MetaLearner**: Manages meta-models (imputation, scaling, encoding)
- **MetaFeatureExtractor**: Extracts meta-features from data columns
- **DecisionLogger**: Tracks all preprocessing decisions
- **DataEvaluator**: Computes data quality metrics

## Supported Strategies

**Imputation**: mean, median, KNN  
**Scaling**: standard, minmax, robust  
**Encoding**: onehot, label

## Requirements

- Python 3.11+
- pandas, scikit-learn, streamlit, plotly
- See `requirements.txt` for full dependencies

## Future Work

- Extended imputation strategies (iterative, domain-specific)
- Feature engineering automation
- Class imbalance handling
- Hyperparameter optimization
- Benchmark comparison with baseline approaches

## License

This project is part of a seminar at FH Schmalkalden University of Applied Sciences.

## Author

Parinaz Teimouri  
Applied Computer Science, FH Schmalkalden

---

For details on methodology, see the papers in `paper/` and `new-paper/` directories.

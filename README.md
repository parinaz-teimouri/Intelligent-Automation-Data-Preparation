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
├── train_meta_model.py       # Script to generate meta-training data
├── test_pipeline.py          # Testing & evaluation script
├── dashboard.py              # Streamlit interactive dashboard
├── input/                    # Raw input datasets
├── meta_models/              # Trained meta-learning models
├── paper/                    # Published paper (LaTeX)
└── new-paper/                # Working paper draft (LaTeX)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Data_Preparation
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Run the pipeline on a CSV file:
```bash
python test_pipeline.py path/to/your/data.csv
```

### Launch the interactive dashboard:
```bash
streamlit run dashboard.py
```

### Train meta-learning models:
```bash
python train_meta_model.py
```

## How It Works

1. **Data Analysis**: Extracts meta-features (missing ratio, skewness, cardinality, etc.)
2. **Strategy Prediction**: Meta-models predict top-K strategies for each preprocessing task
3. **Performance Evaluation**: Evaluates candidates using downstream model cross-validation
4. **Selection**: Chooses the strategy with highest performance
5. **Reporting**: Logs all decisions with detailed change tracking

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

- Python 3.8+
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

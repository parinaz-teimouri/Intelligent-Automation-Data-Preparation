import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from meta_learning import MetaLearner, MetaFeatureExtractor
import os

def generate_synthetic_dataset(n_samples=1000, n_features=10, problem_type='classification', defect_type=None):
    """
    Generates a synthetic dataset with specific defects to test strategies.
    defect_type: 'missing', 'skewed', 'categorical', 'mixed'
    """
    if problem_type == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), random_state=np.random.randint(1000))
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), random_state=np.random.randint(1000))
    
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(n_features)])
    
    # Introduce Defects
    if defect_type == 'missing':
        # Introduce varying degrees of missingness
        mask = np.random.rand(*df.shape) < np.random.uniform(0.05, 0.5)
        df[mask] = np.nan
        best_imputation = 'knn' if n_samples < 5000 else 'mean' # Heuristic labeling
    
    elif defect_type == 'skewed':
        # Exponential transformation
        df = np.exp(df)
        best_scaling = 'robust' # Skewed data needs robust or log
    
    elif defect_type == 'categorical':
        # Convert some to categorical
        for c in df.columns:
            df[c] = pd.cut(df[c], bins=np.random.randint(3, 20), labels=False)
        best_encoding = 'onehot' if df.nunique().mean() < 10 else 'label'

    return df, defect_type

def create_training_data():
    """Generates meta-training data."""
    X_impute, y_impute = [], []
    X_scale, y_scale = [], []
    X_encode, y_encode = [], []
    
    print("Generating synthetic datasets...")
    for _ in range(20):
        # 1. Missing Data Scenarios
        df, _ = generate_synthetic_dataset(defect_type='missing')
        for col in df.columns:
            feats = MetaFeatureExtractor.extract_features(df, col)
            X_impute.append(feats)
            # Simple heuristic labeling for training
            if feats[1] < 0.1: y_impute.append('mean') # Low missing -> Mean
            elif feats[0] < 1000: y_impute.append('knn') # Small data -> KNN
            else: y_impute.append('median') # Default
            
        # 2. Scaling Scenarios
        df, _ = generate_synthetic_dataset(defect_type='skewed')
        for col in df.columns:
            feats = MetaFeatureExtractor.extract_features(df, col)
            X_scale.append(feats)
            if abs(feats[2]) > 1: y_scale.append('robust')
            else: y_scale.append('standard')

        # 3. Encoding Scenarios
        df, _ = generate_synthetic_dataset(defect_type='categorical')
        for col in df.columns:
            feats = MetaFeatureExtractor.extract_features(df, col)
            X_encode.append(feats)
            if feats[4] < 0.05: y_encode.append('onehot')
            else: y_encode.append('label')

    return {
        "imputation": (X_impute, y_impute),
        "scaling": (X_scale, y_scale),
        "encoding": (X_encode, y_encode)
    }

if __name__ == "__main__":
    training_data = create_training_data()
    
    learner = MetaLearner()
    learner.train(training_data)
    print("Meta-models trained successfully.")

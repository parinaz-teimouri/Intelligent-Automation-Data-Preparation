import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class MetaFeatureExtractor:
    @staticmethod
    def extract_features(df, col_name):
        """Extracts meta-features from a column for strategy prediction."""
        series = df[col_name]
        
        # Basic Statistics
        n_samples = len(series)
        missing_ratio = series.isnull().mean()
        
        # Handling numeric vs categorical
        if pd.api.types.is_numeric_dtype(series):
            skewness = series.skew()
            kurtosis = series.kurtosis()
            unique_ratio = series.nunique() / n_samples
            dtype_val = 0 # Numeric
        else:
            skewness = 0
            kurtosis = 0
            unique_ratio = series.nunique() / n_samples
            dtype_val = 1 # Categorical

        return [n_samples, missing_ratio, skewness, kurtosis, unique_ratio, dtype_val]

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_models")

# Candidate strategies per task, in the order used to break ties / fill the Top-K list.
STRATEGIES = {
    "imputation": ["mean", "median", "knn"],
    "scaling": ["standard", "robust", "minmax"],
    "encoding": ["onehot", "label"],
}

class MetaLearner:
    def __init__(self, models_dir=None):
        # Resolve relative to this file so the app works from any working directory.
        models_dir = models_dir or DEFAULT_MODELS_DIR
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        self.imputation_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoding_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaling_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.is_fitted = {
            "imputation": False,
            "encoding": False,
            "scaling": False
        }

    def train(self, training_data):
        """
        Trains the meta-models.
        training_data: Dictionary containing 'imputation', 'encoding', 'scaling' datasets.
                       Each dataset should be (X_meta, y_best_strategy).
        """
        for task, (X, y) in training_data.items():
            if len(X) > 0:
                model = getattr(self, f"{task}_model")
                model.fit(X, y)
                self.is_fitted[task] = True
                
                # Save model
                joblib.dump(model, os.path.join(self.models_dir, f"{task}_model.pkl"))
                print(f"Trained and saved {task} meta-model.")

    def load_models(self):
        """Loads pre-trained meta-models."""
        for task in self.is_fitted.keys():
            path = os.path.join(self.models_dir, f"{task}_model.pkl")
            if os.path.exists(path):
                setattr(self, f"{task}_model", joblib.load(path))
                self.is_fitted[task] = True
                print(f"Loaded {task} meta-model.")

    def predict(self, task, meta_features, exclude=()):
        """Predicts the best strategy for a given task."""
        return self.rank_strategies(task, meta_features, exclude)[0]

    def predict_top_k(self, task, meta_features, k=2, exclude=()):
        """Returns the top k most probable strategies."""
        return self.rank_strategies(task, meta_features, exclude)[:k]

    def rank_strategies(self, task, meta_features, exclude=()):
        """
        Ranks all candidate strategies for a task, most promising first.
        Uses the meta-model's probabilities when fitted, otherwise the fallback heuristic
        followed by the remaining strategies. Strategies in `exclude` are never returned.
        """
        if self.is_fitted[task]:
            model = getattr(self, f"{task}_model")
            X = pd.DataFrame([meta_features])
            try:
                probs = model.predict_proba(X)[0]
                # Sort by probability descending
                ranked = [str(s) for s in model.classes_[np.argsort(probs)[::-1]]]
            except AttributeError:
                # If model doesn't support predict_proba
                ranked = [str(model.predict(X)[0])]
        else:
            ranked = [self._fallback_heuristic(task, meta_features)]

        # Fill up with the remaining known strategies so there is always something to compare.
        ranked += [s for s in STRATEGIES.get(task, []) if s not in ranked]
        ranked = [s for s in ranked if s not in exclude]
        if not ranked:
            raise ValueError(f"No {task} strategies left after excluding {sorted(exclude)}")
        return ranked

    def store_feedback(self, task, meta_features, best_strategy):
        """Stores the result of a real-world evaluation for future training."""
        feedback_file = os.path.join(self.models_dir, f"{task}_feedback.csv")
        
        # Flatten meta_features and add label
        data = list(meta_features) + [best_strategy, pd.Timestamp.now().isoformat()]
        
        # Columns: n_samples, missing_ratio, skewness, kurtosis, unique_ratio, dtype_val, best_strategy, timestamp
        # We'll just append as CSV
        
        df = pd.DataFrame([data])
        
        if os.path.exists(feedback_file):
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            # Define header for clarity if needed, but we keep it simple
            df.to_csv(feedback_file, mode='w', header=False, index=False)
        
        # print(f"Feedback stored for {task}.")

    def _fallback_heuristic(self, task, meta_features):
        """Static rules if no model is trained."""
        # meta_features: [n_samples, missing_ratio, skewness, kurtosis, unique_ratio, dtype_val]
        missing_ratio = meta_features[1]
        skewness = meta_features[2]
        unique_ratio = meta_features[4]
        dtype_val = meta_features[5]

        if task == "imputation":
            if dtype_val == 1: return "most_frequent"
            if missing_ratio < 0.1: return "mean"
            return "knn"
        
        elif task == "encoding":
            if unique_ratio < 0.05: return "onehot"
            return "label"
            
        elif task == "scaling":
            if abs(skewness) > 1: return "robust"
            return "standard"
            
        return "default"

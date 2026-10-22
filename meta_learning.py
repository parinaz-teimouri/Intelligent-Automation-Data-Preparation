import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Names of the meta-features returned by MetaFeatureExtractor.extract_features, in order.
FEATURE_NAMES = ["n_samples", "missing_ratio", "skewness", "kurtosis", "unique_ratio", "is_categorical"]
FEEDBACK_COLUMNS = FEATURE_NAMES + ["best_strategy", "timestamp"]

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
            # Constant or tiny columns have undefined skew/kurtosis; treat them as symmetric
            skewness = 0.0 if pd.isna(series.skew()) else series.skew()
            kurtosis = 0.0 if pd.isna(series.kurtosis()) else series.kurtosis()
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
                model.fit(pd.DataFrame(X, columns=FEATURE_NAMES), list(y))
                self.is_fitted[task] = True

                # Save model
                joblib.dump(model, os.path.join(self.models_dir, f"{task}_model.pkl"))
                print(f"Trained and saved {task} meta-model.")

    def load_models(self):
        """Loads pre-trained meta-models."""
        for task in self.is_fitted.keys():
            path = os.path.join(self.models_dir, f"{task}_model.pkl")
            if os.path.exists(path):
                model = joblib.load(path)
                # Models trained on a different meta-feature layout would silently mispredict
                if list(getattr(model, "feature_names_in_", [])) != FEATURE_NAMES:
                    print(f"Ignoring outdated {task} meta-model (retrain with train_meta_model.py).")
                    continue
                setattr(self, f"{task}_model", model)
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
            X = pd.DataFrame([meta_features], columns=FEATURE_NAMES)
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

    def _feedback_path(self, task):
        return os.path.join(self.models_dir, f"{task}_feedback.csv")

    def store_feedback(self, task, meta_features, best_strategy):
        """Stores the result of a real-world evaluation for future training."""
        feedback_file = self._feedback_path(task)
        row = pd.DataFrame([list(meta_features) + [best_strategy, pd.Timestamp.now().isoformat()]],
                           columns=FEEDBACK_COLUMNS)

        if os.path.exists(feedback_file):
            with open(feedback_file) as f:
                header = f.readline().strip().split(",")
            if header == FEEDBACK_COLUMNS:
                row.to_csv(feedback_file, mode='a', header=False, index=False)
                return
            self._archive_feedback(feedback_file)

        row.to_csv(feedback_file, mode='w', header=True, index=False)

    def _archive_feedback(self, feedback_file):
        """Moves a feedback file with an outdated layout to legacy/ so it is not mixed with new rows."""
        legacy_dir = os.path.join(self.models_dir, "legacy")
        os.makedirs(legacy_dir, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(feedback_file))
        stamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        os.replace(feedback_file, os.path.join(legacy_dir, f"{name}_{stamp}{ext}"))
        print(f"Archived outdated feedback file {feedback_file} to {legacy_dir}.")

    def load_feedback(self, task):
        """Returns (X_meta, y_best_strategy) collected from evaluated pipeline runs."""
        feedback_file = self._feedback_path(task)
        if not os.path.exists(feedback_file):
            return [], []
        feedback = pd.read_csv(feedback_file)
        if list(feedback.columns) != FEEDBACK_COLUMNS:
            return [], []
        return feedback[FEATURE_NAMES].values.tolist(), feedback["best_strategy"].tolist()

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

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import os
import re
import json
import warnings
from datetime import datetime
from meta_learning import MetaLearner, MetaFeatureExtractor

# Value standardisation
TRUE_TOKENS = {"true", "yes", "y", "t", "1", "ja", "wahr"}
FALSE_TOKENS = {"false", "no", "n", "f", "0", "nein", "falsch"}
# Currency markers and percent signs around a number, e.g. "$75", "USD 30", "89,90 €", "84%"
NUMBER_DECORATION = re.compile(r"^\s*(?:USD|EUR|GBP|CHF|\$|€|£)\s*|\s*(?:USD|EUR|GBP|CHF|\$|€|£|%)\s*$", re.IGNORECASE)
# Number formats: "89,90" (decimal comma), "1.299,00" (German thousands), "1,200.50" (English thousands)
DECIMAL_COMMA = re.compile(r"-?\d+,\d{1,2}")
GERMAN_THOUSANDS = re.compile(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?")
ENGLISH_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
NUMERIC_PARSE_THRESHOLD = 0.9
# Dates need a separator between digit groups ("2021-05-12", "12/05/2021")
DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}")
# "01.12.2010" is day-first, as written in Germany
DOTTED_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{2,4}")
IDENTIFIER_UNIQUE_RATIO = 0.95
# Name tokens that mark an identifier column ("customer_id", "CustomerID", "InvoiceNo", "sku")
ID_NAME_TOKENS = {"id", "uuid", "sku", "ean", "gtin"}
ID_NAME_SUFFIXES = {"no", "nr"}
# Single-letter abbreviations ('m' -> 'male') are only merged in columns with at most this many categories
ABBREVIATION_MAX_CATEGORIES = 10
# Categorical columns missing more than this share get an explicit "(missing)" category instead of a guess
MISSING_CATEGORY_THRESHOLD = 0.1
MISSING_CATEGORY = "(missing)"


def _normalise_number(text, decimal_comma):
    """Rewrites one number in German or English notation into a plain '1234.5' form."""
    if DECIMAL_COMMA.fullmatch(text):
        return text.replace(",", ".")
    # "1.299" alone is ambiguous: German thousands only when the column also uses decimal commas
    if GERMAN_THOUSANDS.fullmatch(text) and ("," in text or decimal_comma):
        return text.replace(".", "").replace(",", ".")
    if ENGLISH_THOUSANDS.fullmatch(text):
        return text.replace(",", "")
    return text


def _parse_numeric_text(series):
    """Converts text like '$75', '89,90 €', 'USD 1,200' or '84%' to numbers (NaN where not parseable)."""
    cleaned = series.astype(str).str.strip().str.replace(NUMBER_DECORATION, "", regex=True).str.strip()
    decimal_comma = bool(cleaned.str.fullmatch(DECIMAL_COMMA.pattern).any())
    normalised = cleaned.map(lambda text: _normalise_number(text, decimal_comma))
    return pd.to_numeric(normalised, errors="coerce").where(series.notna())


def _category_mapping(series):
    """
    Maps spelling variants of the same category onto one canonical label.
    Variants differing only in case/whitespace are grouped; in small categorical columns a
    single-letter value joins the only longer value starting with that letter (e.g. 'f' -> 'female').
    The form used most overall wins (sizes keep 'M' when 'M'/'m' outnumber 'medium'; gender keeps
    'Male' when 'Male'/'MALE'/'male' outnumber 'M'/'m'), written in its most frequent spelling.
    Returns {original: canonical} for values that change.
    """
    values = series.dropna().astype(str)
    stripped = values.str.strip()
    keys = stripped.str.replace(r"\s+", " ", regex=True).str.casefold()
    spellings = {key: group.value_counts() for key, group in stripped.groupby(keys)}

    # In code-like columns with many values (e.g. ticker symbols) a single letter is a value of its own.
    merged_into = {}
    if len(spellings) <= ABBREVIATION_MAX_CATEGORIES:
        for key in spellings:
            if len(key) == 1:
                longer = [k for k in spellings if len(k) > 1 and k.startswith(key)]
                if len(longer) == 1:
                    merged_into[key] = longer[0]

    members = {}  # group -> keys merged into it
    for key in spellings:
        members.setdefault(merged_into.get(key, key), []).append(key)
    canonical = {}
    for group, keys_in_group in members.items():
        # Ties resolve alphabetically so results are reproducible
        winner = min(keys_in_group, key=lambda k: (-spellings[k].sum(), k))
        counts = spellings[winner]
        canonical[group] = min(counts.index, key=lambda s: (-counts[s], s))

    return {original: canonical[merged_into.get(key, key)] for original, key in zip(values, keys)
            if original != canonical[merged_into.get(key, key)]}


def _name_tokens(name):
    """'CustomerID' / 'customer_id' / 'Invoice No' -> ['customer', 'id'] etc."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token]


def _is_identifier(name, series):
    """
    Identifier / free-text columns carry nothing that generalises:
    text where (almost) every value is unique, or columns named like an ID with many distinct values
    (a customer ID repeats across orders but is still a key, not a measurement).
    """
    non_null = series.dropna()
    if len(non_null) < 10:
        return False
    n_unique = non_null.nunique()
    unique_ratio = n_unique / len(non_null)
    if series.dtype == "object" and unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
        return True

    tokens = _name_tokens(name)
    named_like_id = bool(tokens) and (bool(ID_NAME_TOKENS & set(tokens)) or tokens[-1] in ID_NAME_SUFFIXES)
    if not named_like_id:
        return False
    if pd.api.types.is_numeric_dtype(series) and not np.allclose(non_null, np.round(non_null)):
        return False
    # A handful of repeating values ("store_id" with 5 stores) is a real category
    return unique_ratio >= IDENTIFIER_UNIQUE_RATIO or n_unique >= max(20, 0.05 * len(non_null))


def make_imputer(strategy):
    """Returns the imputer for a numeric imputation strategy name."""
    if strategy == "knn": return KNNImputer(n_neighbors=5)
    if strategy in ("mean", "median", "most_frequent"): return SimpleImputer(strategy=strategy)
    raise ValueError(f"Unknown imputation strategy: {strategy}")


def make_scaler(strategy):
    """Returns the scaler for a scaling strategy name."""
    if strategy == "standard": return StandardScaler()
    if strategy == "minmax": return MinMaxScaler()
    if strategy == "robust": return RobustScaler()
    raise ValueError(f"Unknown scaling strategy: {strategy}")


def make_encoder(strategy):
    """Returns the sklearn encoder for an encoding strategy name (used for evaluation)."""
    if strategy == "onehot": return OneHotEncoder(handle_unknown="ignore")
    if strategy == "label": return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unknown encoding strategy: {strategy}")


def _as_str(X):
    # Categorical columns can mix types (e.g. 1 and "Y"); the encoder needs one type.
    return X.astype(str)


# --- Candidate evaluation --------------------------------------------------------
# Shared by the pipeline (Top-K optimisation) and train_meta_model.py (meta-training labels),
# so both judge strategies the same way.

def evaluation_model_for(task, task_type):
    """
    Downstream model used to score candidates. Tree models ignore feature scale,
    so scalers are compared with a scale-sensitive k-NN model instead.
    """
    regression = task_type == "regression"
    if task == "scaling":
        return KNeighborsRegressor(n_neighbors=5) if regression else KNeighborsClassifier(n_neighbors=5)
    if regression:
        return RandomForestRegressor(n_estimators=50, random_state=42)
    return RandomForestClassifier(n_estimators=50, random_state=42)


def cv_splitter(y, task_type):
    """Stratified folds for classification, reduced so every class appears in each fold."""
    if task_type == "classification":
        n_splits = min(3, int(pd.Series(y).value_counts().min()))
        if n_splits >= 2:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=min(3, len(y)), shuffle=True, random_state=42)


def build_eval_pipeline(task, strategy, X, task_type, scale_cols=None):
    """
    Builds the full preprocessing + model pipeline used to score one candidate.
    The candidate is plugged into its own step; the other steps use safe defaults.
    """
    model = evaluation_model_for(task, task_type)

    if task == "scaling":
        # Features are already imputed and encoded; scale the same columns the pipeline would.
        preprocess = ColumnTransformer([("scale", make_scaler(strategy), scale_cols)], remainder="passthrough")
    else:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        imputer = make_imputer(strategy if task == "imputation" else "median")
        encoder = make_encoder(strategy if task == "encoding" else "onehot")
        numeric = Pipeline([("imputer", imputer), ("scaler", StandardScaler())])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("as_str", FunctionTransformer(_as_str)),
            ("encoder", encoder),
        ])
        preprocess = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate_strategy(task, strategy, X, y, task_type, scale_cols=None):
    """
    Cross-validates one candidate strategy with a downstream model.
    Returns (score, error): score is F1 (classification) or R2 (regression),
    error is the failure message if the evaluation could not run.
    """
    scoring = "r2" if task_type == "regression" else "f1_macro"
    try:
        pipeline = build_eval_pipeline(task, strategy, X, task_type, scale_cols)
        scores = cross_val_score(pipeline, X, y, cv=cv_splitter(y, task_type), scoring=scoring, error_score="raise")
        return float(scores.mean()), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

class DecisionLogger:
    def __init__(self):
        self.logs = []
        self.changes = []  # Track specific data changes
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "steps_taken": []
        }

    def log(self, step, decision, reason, details=None):
        entry = {
            "step": step,
            "decision": decision,
            "reason": reason,
            "details": details or {}
        }
        self.logs.append(entry)
        self.metadata["steps_taken"].append(step)
        print(f"[{step}] {decision} - {reason}")

    def log_change(self, category, column, old_value_desc, new_value_desc, affected_count):
        """Logs a specific data transformation with before/after details."""
        change_entry = {
            "category": category,
            "column": column,
            "before": old_value_desc,
            "after": new_value_desc,
            "affected_rows": affected_count
        }
        self.changes.append(change_entry)

    def save_report(self, output_path):
        report = {
            "metadata": self.metadata,
            "changes_summary": self.changes,
            "decisions": self.logs
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nReport saved to {output_path}")

class DataEvaluator:
    @staticmethod
    def calculate_quality_metrics(df):
        if df is None: return {}
        
        metrics = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_percentage": float(df.isnull().mean().mean() * 100),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_types": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
            "memory_usage_kb": float(df.memory_usage(deep=True).sum() / 1024)
        }
        
        # Calculate outlier estimate (simple IQR based for summary)
        num_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_count += ((df[col] < lower) | (df[col] > upper)).sum()
        metrics["estimated_outliers"] = int(outlier_count)
        
        return metrics


class AutoDataPreparer:
    # KNN imputation is quadratic in the number of rows, so it is pruned from the candidates above this size.
    KNN_MAX_ROWS = 10_000

    def __init__(self, file_path, target_col=None, models_dir=None, fixed_strategies=None,
                 standardize=True, optimize=True, record_feedback=True):
        """
        fixed_strategies: e.g. {"imputation": "mean", "scaling": "standard", "encoding": "onehot"}
                          to skip strategy selection for those tasks (static pipelines, benchmarks).
        standardize:      repair text values and exclude identifier columns.
        optimize:         score candidates against the target (otherwise strategies are only predicted).
        record_feedback:  store evaluation winners for future meta-training.
        """
        self.file_path = file_path
        self.fixed_strategies = fixed_strategies or {}
        self.standardize = standardize
        self.optimize = optimize
        self.record_feedback = record_feedback
        self.chosen_strategies = {} # task -> strategy actually applied ("encoding": {column: strategy})
        self.data = None
        self.raw_data_snapshot = None # Keep a copy for evaluation
        self.logger = DecisionLogger()
        self.target_col = target_col
        self.target = None # Target series, kept apart from the features during preprocessing
        self.identifiers = None # ID / free-text columns, kept for the human-readable output only
        self.integer_columns = set() # Columns whose original values are whole numbers
        self.task_type = "classification" # Default
        self.start_time = None
        
        # Initialize Meta-Learner
        self.meta_learner = MetaLearner(models_dir)
        self.meta_learner.load_models()
        
    def _separate_target(self):
        """
        Removes the target column from the features so that no preprocessing step
        (capping, imputation, scaling) or candidate evaluation can touch or see it.
        Rows without a target value are dropped: they cannot be used for supervised evaluation.
        """
        if not self.target_col or self.data is None:
            return
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        missing = self.data[self.target_col].isnull()
        n_missing = int(missing.sum())
        if n_missing > 0:
            self.data = self.data[~missing].copy()
            self.logger.log("Target", "Dropped Rows", "Rows without a target value cannot be used for supervised evaluation",
                           {"column": self.target_col, "count": n_missing})
            self.logger.log_change("Target", self.target_col, "NaN", "Row dropped", n_missing)

        self.target = self.data.pop(self.target_col)

    def _align_target(self):
        """Keeps the target and identifiers aligned with the feature rows after rows are dropped."""
        if self.target is not None:
            self.target = self.target.loc[self.data.index]
        if self.identifiers is not None:
            self.identifiers = self.identifiers.loc[self.data.index]

    def _with_target(self):
        """Returns the features with the target column re-attached as the last column."""
        if self.target is None:
            return self.data
        return pd.concat([self.data, self.target], axis=1)

    def _human_readable_frame(self):
        """Identifiers, features and target: everything a person needs to trace a row."""
        if self.identifiers is None:
            return self._with_target()
        return pd.concat([self.identifiers, self._with_target()], axis=1)

    def standardize_values(self):
        """
        Repairs inconsistent text values in feature columns:
        boolean-like text -> 0/1, numbers stored as text -> numeric, and spelling
        variants of the same category -> one label.
        """
        if self.data is None: return

        for col in self.data.select_dtypes(include=['object']).columns:
            series = self.data[col]
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            tokens = non_null.astype(str).str.strip().str.casefold()

            # 1. Booleans written as text ("Y", "False", "1", ...)
            if set(tokens) <= TRUE_TOKENS | FALSE_TOKENS:
                self.data[col] = tokens.map(lambda t: 1.0 if t in TRUE_TOKENS else 0.0).reindex(series.index)
                self.logger.log("Standardization", "Boolean Text to 0/1", f"Column {col} holds yes/no style values",
                                {"column": col, "values": sorted(set(non_null.astype(str)))})
                self.logger.log_change("Standardization", col, "Yes/No style text", "0/1", len(non_null))
                continue

            # 2. Numbers stored as text ("$75", "USD 30", "84%")
            parsed = _parse_numeric_text(series)
            parse_rate = parsed.notna().sum() / len(non_null)
            if parse_rate >= NUMERIC_PARSE_THRESHOLD:
                unparseable = int(len(non_null) - parsed.notna().sum())
                self.data[col] = parsed
                self.logger.log("Standardization", "Text to Number", f"Column {col} holds numbers stored as text",
                                {"column": col, "parse_rate": f"{parse_rate:.1%}", "unparseable_set_to_missing": unparseable})
                self.logger.log_change("Standardization", col, "Numeric text (currency/percent)", "Number", len(non_null))
                continue

            # 3. Spelling variants of the same category ("Male", "MALE", "m")
            mapping = _category_mapping(series)
            if mapping:
                affected = int(series.isin(list(mapping)).sum())
                self.data[col] = series.replace(mapping)
                self.logger.log("Standardization", "Merged Category Variants", f"Column {col} had inconsistent spellings",
                                {"column": col, "mapping": mapping})
                self.logger.log_change("Standardization", col, f"Variants: {sorted(mapping)[:5]}",
                                       f"Merged into {sorted(set(mapping.values()))}", affected)

    def _separate_identifiers(self):
        """Moves ID and free-text columns out of the features; they stay in the human-readable file."""
        if self.data is None: return
        id_cols = [col for col in self.data.columns if _is_identifier(col, self.data[col])]
        if not id_cols:
            return
        self.identifiers = self.data[id_cols]
        self.data = self.data.drop(columns=id_cols)
        self.logger.log("Feature Selection", "Excluded Identifiers",
                        "Columns with (almost) all-unique values carry no generalisable signal",
                        {"columns": id_cols})

    def _record_integer_columns(self):
        """Remembers which numeric columns hold whole numbers before imputation/capping adds fractions."""
        for col in self.data.select_dtypes(include=[np.number]).columns:
            non_na = self.data[col].dropna()
            if len(non_na) > 0 and np.allclose(non_na, np.round(non_na)):
                self.integer_columns.add(col)

    def _detect_task_type(self):
        """
        Detects if the problem is Classification or Regression based on target column.
        """
        if self.target is None:
            return

        target_series = self.target
        
        # Heuristic:
        # 1. If non-numeric or boolean -> Classification
        # 2. If numeric with fractional values (e.g. a 3.3 rating) -> Regression
        # 3. If integer-valued with low cardinality (<20 unique or <5% unique) -> Classification
        # 4. Else -> Regression

        integer_valued = None
        if not pd.api.types.is_numeric_dtype(target_series) or pd.api.types.is_bool_dtype(target_series):
            self.task_type = "classification"
        else:
            integer_valued = bool(np.allclose(target_series, np.round(target_series)))
            unique_count = target_series.nunique()
            total_count = len(target_series)
            if integer_valued and (unique_count < 20 or (unique_count / total_count < 0.05)):
                self.task_type = "classification"
            else:
                self.task_type = "regression"

        self.logger.log("Task Detection", self.task_type.title(), f"Detected based on target '{self.target_col}' properties",
                       {"target_dtype": str(target_series.dtype), "unique_values": int(target_series.nunique()),
                        "integer_valued": integer_valued})

    def load_data(self):
        """Loads data from CSV or Excel files."""
        _, ext = os.path.splitext(self.file_path)
        try:
            if ext.lower() == '.csv':
                self.data = pd.read_csv(self.file_path)
            elif ext.lower() in ['.xls', '.xlsx']:
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            self.raw_data_snapshot = self.data.copy()
            self.logger.log("Load Data", "Success", f"Loaded file {self.file_path}", 
                          {"shape": self.data.shape, "columns": list(self.data.columns)})
            
        except Exception as e:
            self.logger.log("Load Data", "Failed", str(e))
            raise

    def analyze_data(self):
        """Basic data analysis."""
        if self.data is None: return
        # print("\n--- Data Analysis ---")
        # print(self.data.info())
        # print("\nMissing Values:\n", self.data.isnull().sum())
        # print("\nDuplicates:", self.data.duplicated().sum())

    def remove_duplicates(self):
        """Removes exact duplicate rows (a row is only a duplicate if its target matches too)."""
        if self.data is None: return

        initial_rows = len(self.data)
        duplicated = self._human_readable_frame().duplicated()
        self.data = self.data[~duplicated]
        self._align_target()
        dropped = initial_rows - len(self.data)
        if dropped > 0:
            self.logger.log("Deduplication", "Removed Duplicates", "Exact duplicate rows found", {"count": dropped})
        else:
            self.logger.log("Deduplication", "Skipped", "No duplicates found")

    def clean_data(self):
        """Handles outliers dynamically."""
        if self.data is None: return

        # Outlier Detection & Handling (Dynamic Approach)
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            # print("\nAnalyzing outliers dynamically...")
            for col in num_cols:
                # Use robust statistics (median/IQR) to avoid being skewed by outliers themselves
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Dynamic threshold: If IQR is 0 (constant values), skip
                if IQR == 0:
                    continue
                    
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Check outlier percentage
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound))
                outlier_ratio = outliers.mean()
                
                # Intelligent Decision:
                # If outliers are rare (<5%), cap them (winsorization).
                # If outliers are common (>5%), consider it a heavy-tailed distribution and leave them (or suggest log transform).
                if 0 < outlier_ratio < 0.05:
                    outlier_count = outliers.sum()
                    self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                    self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
                    self.logger.log("Outlier Handling", "Capping Applied", f"Rare outliers detected in {col}", 
                                  {"column": col, "ratio": f"{outlier_ratio:.1%}", "action": "Winsorization"})
                    self.logger.log_change("Outlier Capping", col, 
                                           f"Values outside [{lower_bound:.2f}, {upper_bound:.2f}]", 
                                           "Capped to boundaries", int(outlier_count))
                elif outlier_ratio >= 0.05:
                    self.logger.log("Outlier Handling", "Kept As-Is", f"High outlier count in {col}, likely natural distribution", 
                                  {"column": col, "ratio": f"{outlier_ratio:.1%}", "action": "None"})

    def _evaluate_candidate_strategy(self, task, strategy, X, y, scale_cols=None):
        """Evaluates a specific strategy using a downstream model. Returns (score, error)."""
        return evaluate_strategy(task, strategy, X, y, self.task_type, scale_cols)

    def _excluded_strategies(self, task):
        """Strategies pruned up front because they are too expensive for this dataset."""
        if task == "imputation" and len(self.data) > self.KNN_MAX_ROWS:
            self.logger.log("Imputation", "KNN Excluded",
                            f"{len(self.data)} rows exceeds {self.KNN_MAX_ROWS}; KNN imputation scales quadratically")
            return {"knn"}
        return set()

    def _optimize_strategy(self, task, step, meta_features, X, y, scale_cols=None):
        """
        Top-K optimisation: the meta-learner proposes candidates, each is scored with a
        downstream model, and the best one wins. Feedback is only stored for real evaluations.
        """
        candidates = self.meta_learner.predict_top_k(task, meta_features, k=2,
                                                     exclude=self._excluded_strategies(task))
        self.logger.log(step, "Optimizing...", f"Evaluating Top-K: {candidates}")

        metric_name = "R2" if self.task_type == "regression" else "F1"
        evaluation_model = type(evaluation_model_for(task, self.task_type)).__name__
        scores = {}
        best_strategy, best_score = None, -np.inf

        for strategy in candidates:
            score, error = self._evaluate_candidate_strategy(task, strategy, X, y, scale_cols)
            if error:
                scores[strategy] = None
                self.logger.log("Evaluation", f"{strategy} (Failed)", error)
                continue
            scores[strategy] = score
            self.logger.log("Evaluation", f"{strategy}", f"{metric_name} Score: {score:.4f}")
            if score > best_score:
                best_score, best_strategy = score, strategy

        details = {"candidates": candidates, "scores": scores, "metric": metric_name,
                   "evaluation_model": evaluation_model}

        if best_strategy is None:
            strategy = candidates[0]
            self.logger.log(step, f"{strategy.title()} (Fallback)",
                            "All candidate evaluations failed; using the meta-model's top prediction", details)
            return strategy

        details["best_score"] = best_score
        self.logger.log(step, f"{best_strategy.title()} (Winner)", f"Selected after evaluation (Score: {best_score:.4f})", details)

        # Feedback Loop
        if self.record_feedback:
            self.meta_learner.store_feedback(task, meta_features, best_strategy)
        return best_strategy

    def _choose_strategy(self, task, step, meta_features, X=None, scale_cols=None):
        """Fixed by configuration, optimised against the target, or predicted by the meta-model."""
        if task in self.fixed_strategies:
            strategy = self.fixed_strategies[task]
            self.logger.log(step, f"{strategy.title()} (Fixed)", "Strategy set by configuration")
        elif self.optimize and self.target is not None and X is not None:
            strategy = self._optimize_strategy(task, step, meta_features, X, self.target, scale_cols)
        else:
            # FAST MODE (Prediction only)
            strategy = self.meta_learner.predict(task, meta_features, exclude=self._excluded_strategies(task))
            self.logger.log(step, f"{strategy.title()} Strategy", "Predicted by Meta-Model", {"predicted_strategy": strategy})
        return strategy

    def handle_missing_values(self):
        """Handles missing values using Meta-Learning with Top-K Optimization."""
        if self.data is None: return
        
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Binary 0/1 columns are filled with their most frequent value: a mean of 0.67 is not a valid flag.
        binary_cols = [c for c in num_cols
                       if self.data[c].isnull().any() and set(self.data[c].dropna().unique()) <= {0, 1}]
        if binary_cols:
            for col in binary_cols:
                self.logger.log_change("Imputation", col, "NaN", "Most Frequent Value (binary)", int(self.data[col].isnull().sum()))
            self.data[binary_cols] = SimpleImputer(strategy='most_frequent').fit_transform(self.data[binary_cols])
            self.logger.log("Imputation", "Most Frequent", "Binary 0/1 columns (Default)", {"columns": binary_cols})

        if len(num_cols) > 0:
            cols_with_missing = [c for c in num_cols if self.data[c].isnull().sum() > 0]

            if cols_with_missing:
                worst_col = self.data[cols_with_missing].isnull().sum().idxmax()
                meta_features = MetaFeatureExtractor.extract_features(self.data, worst_col)
                
                # Decision Logic: candidates are scored on all features
                # (date columns are dropped later, so they are not part of the evaluation)
                X_eval = self.data.select_dtypes(exclude=['datetime64', 'datetime64[ns]'])
                strategy = self._choose_strategy("imputation", "Imputation", meta_features, X_eval)
                self.chosen_strategies["imputation"] = strategy

                # Apply Winner
                missing_mask = self.data[num_cols].isnull()
                for col in num_cols:
                    n_missing = missing_mask[col].sum()
                    if n_missing > 0:
                        self.logger.log_change("Imputation", col, "NaN", f"Filled using {strategy}", int(n_missing))

                imputer = make_imputer(strategy)
                self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
        
        # Categorical Imputation: a few gaps get the most frequent value; many gaps get an explicit
        # "(missing)" category, since copying one value into most rows would invent a pattern.
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            frequent_cols, missing_cols = [], []
            for col in cat_cols:
                n_missing = int(self.data[col].isnull().sum())
                if n_missing == 0:
                    frequent_cols.append(col)
                elif n_missing / len(self.data) > MISSING_CATEGORY_THRESHOLD:
                    missing_cols.append(col)
                    self.data[col] = self.data[col].astype(object).where(self.data[col].notna(), MISSING_CATEGORY)
                    self.logger.log_change("Imputation", col, "NaN", f"Category '{MISSING_CATEGORY}'", n_missing)
                else:
                    frequent_cols.append(col)
                    self.logger.log_change("Imputation", col, "NaN", "Most Frequent Value", n_missing)

            if frequent_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                self.data[frequent_cols] = cat_imputer.fit_transform(self.data[frequent_cols])
                self.logger.log("Imputation", "Most Frequent", "Categorical columns (Default)", {"columns": frequent_cols})
            if missing_cols:
                self.logger.log("Imputation", "Missing Category",
                                f"More than {MISSING_CATEGORY_THRESHOLD:.0%} missing: marked as '{MISSING_CATEGORY}' instead of guessed",
                                {"columns": missing_cols})

    def parse_dates(self):
        """Attempts to parse date columns safely."""
        if self.data is None: return
        
        for col in self.data.columns:
            # Only check object columns that are NOT already numeric
            if self.data[col].dtype == 'object':
                # Safe Check: Sample the column to see if it looks like a date
                non_na_sample = self.data[col].dropna().head(100)
                if len(non_na_sample) == 0: continue

                # Only consider values shaped like dates (digit groups with a separator)
                sample_text = non_na_sample.astype(str)
                if sample_text.str.contains(DATE_LIKE).mean() <= 0.8:
                    continue
                # German "01.12.2010" means 1 December
                dayfirst = bool(sample_text.str.match(DOTTED_DATE).mean() > 0.8)

                try:
                    # We check if >80% of non-null values can be parsed
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)  # "could not infer format" noise
                        parsed = pd.to_datetime(non_na_sample, errors='coerce', dayfirst=dayfirst)
                    success_rate = parsed.notna().mean()

                    if success_rate > 0.8:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            self.data[col] = pd.to_datetime(self.data[col], errors='coerce', dayfirst=dayfirst)
                        
                        # Extract useful features
                        self.data[f'{col}_year'] = self.data[col].dt.year
                        self.data[f'{col}_month'] = self.data[col].dt.month
                        self.data[f'{col}_day'] = self.data[col].dt.day
                        self.logger.log("Feature Engineering", "Date Parsing", f"Identified date column: {col}", 
                                      {"new_features": [f'{col}_year', f'{col}_month', f'{col}_day']})
                except Exception:
                    pass

    def enforce_types_and_rounding(self):
        """Enforces integer types for specific columns and rounds values where appropriate."""
        if self.data is None: return

        # Columns whose original values were all whole numbers (age, counts, year/month/day, flags)
        # are rounded back after imputation or capping introduced fractions.
        for col in self.data.columns:
            if col in self.integer_columns and pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = self.data[col].round()
                try:
                    self.data[col] = self.data[col].astype('Int64')
                except (TypeError, ValueError):
                    pass

    def encode_categorical(self):
        """Encodes categorical variables using Meta-Learning."""
        if self.data is None: return
        
        # Drop date columns before encoding/ML processing
        date_cols = self.data.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
        if len(date_cols) > 0:
            self.data.drop(columns=date_cols, inplace=True)
            self.logger.log("Preprocessing", "Drop Dates", "Dropped original date columns for ML compatibility", {"columns": list(date_cols)})

        cat_cols = self.data.select_dtypes(include=['object', 'category', 'bool']).columns
        if len(cat_cols) > 0:
            total_rows = len(self.data)
            for col in cat_cols:
                # Boolean handling: Ensure 0/1
                if self.data[col].dtype == 'bool':
                    self.data[col] = self.data[col].astype(int)
                    self.logger.log("Encoding", "Boolean to Int", f"Converted {col} to 0/1", {"column": col})
                    continue

                # Meta-Learning Decision
                if "encoding" in self.fixed_strategies:
                    strategy = self.fixed_strategies["encoding"]
                else:
                    meta_features = MetaFeatureExtractor.extract_features(self.data, col)
                    strategy = self.meta_learner.predict("encoding", meta_features)
                self.chosen_strategies.setdefault("encoding", {})[col] = strategy

                if strategy == "onehot":
                    dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                    new_cols = list(dummies.columns)
                    self.data = pd.concat([self.data, dummies.astype(int)], axis=1)
                    self.data.drop(columns=[col], inplace=True)
                    self.logger.log("Encoding", "One-Hot Encoding", f"Predicted by Meta-Model", 
                                  {"column": col})
                    self.logger.log_change("Encoding", col, "Categorical Column", f"Expanded into {len(new_cols)} columns: {new_cols[:5]}...", len(self.data))
                else: # Label
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.logger.log("Encoding", "Label Encoding", f"Predicted by Meta-Model", 
                                  {"column": col})
                    self.logger.log_change("Encoding", col, "Categorical Strings", "Encoded to Integers", len(self.data))

    def encode_target(self):
        """Label encodes a non-numeric classification target; numeric targets are left untouched."""
        if self.target is None or pd.api.types.is_numeric_dtype(self.target):
            return

        le = LabelEncoder()
        self.target = pd.Series(le.fit_transform(self.target.astype(str)), index=self.target.index, name=self.target.name)
        mapping = {str(label): int(code) for code, label in enumerate(le.classes_)}
        self.logger.log("Encoding", "Target Encoding", "Label Encoded Target Variable",
                        {"column": self.target_col, "mapping": mapping})

    def scale_features(self):
        """Scales numerical features using Meta-Learning with Top-K Optimization."""
        if self.data is None: return
        
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Filter out boolean/binary columns (0/1) from scaling
        cols_to_scale = [col for col in num_cols if self.data[col].nunique() > 2]
        
        if len(cols_to_scale) > 0:
            
            # Find column with max absolute skew
            skews = self.data[cols_to_scale].skew().abs()
            max_skew_col = skews.idxmax()
            meta_features = MetaFeatureExtractor.extract_features(self.data, max_skew_col)
            
            # Decision Logic
            # Data is already encoded and numeric now; the target is held separately
            strategy = self._choose_strategy("scaling", "Scaling", meta_features, self.data, scale_cols=cols_to_scale)
            self.chosen_strategies["scaling"] = strategy

            # Apply Winner
            scaler = make_scaler(strategy)

            for col in cols_to_scale:
                old_min, old_max = self.data[col].min(), self.data[col].max()
                self.logger.log_change("Scaling", col, f"Range [{old_min:.2f}, {old_max:.2f}]", f"Scaled using {strategy}", len(self.data))
                
            self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])

    def save_data(self, output_path, human_readable=False):
        """Saves the processed data (features plus target, and identifiers if human readable)."""
        if self.data is None: return
        frame = self._human_readable_frame() if human_readable else self._with_target()
        frame.to_csv(output_path, index=False)
        # print(f"\nProcessed data saved to {output_path}")

    def run_pipeline(self, output_path='processed_data.csv'):
        """Runs the full preparation pipeline."""
        self.start_time = datetime.now()
        print("--- Pipeline Started ---")
        self.load_data()
        
        # Initial Evaluation
        initial_metrics = DataEvaluator.calculate_quality_metrics(self.data)
        
        self.analyze_data()
        self._separate_target() # Keep the target out of all feature preprocessing
        self._detect_task_type() # Detect Classification vs Regression
        if self.standardize:
            self.standardize_values() # Booleans, numbers stored as text, category spellings
        self.parse_dates() # Safer Date Parsing
        self.remove_duplicates() # After standardisation, so 'Male' and 'male' rows can match
        if self.standardize:
            self._separate_identifiers() # IDs / free text stay out of the ML features
        self._record_integer_columns()
        self.clean_data() # Dynamic Outlier Handling

        # Pipeline Level Log
        self.logger.log("Pipeline Info", "Optimization Strategy", "Steps are currently optimized independently (Imputation -> Encoding -> Scaling)")
        
        self.handle_missing_values() # Dynamic Strategy
        self.enforce_types_and_rounding()
        
        # Save Human Readable Version
        dirname, basename = os.path.split(output_path)
        if basename.startswith('processed_'):
            human_basename = basename.replace('processed_', 'human_readable_', 1)
        else:
            human_basename = f'human_readable_{basename}'
        
        human_readable_path = os.path.join(dirname, human_basename)
        self.save_data(human_readable_path, human_readable=True)
        self.logger.log("Output", "Save Human Readable", "Saved intermediate cleaned data", {"path": human_readable_path})

        self.encode_categorical() # Dynamic Thresholds
        self.encode_target()
        self.scale_features() # Dynamic Scaler Selection
        self.analyze_data() 
        self.save_data(output_path)
        self.logger.log("Output", "Save ML Ready", "Saved final processed data", {"path": output_path})
        
        # Final Evaluation
        final_metrics = DataEvaluator.calculate_quality_metrics(self._with_target())
        
        # Save Report
        report_path = os.path.join(dirname, f'report_{os.path.splitext(basename)[0]}.json')
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.logger.metadata["evaluation"] = {
            "before": initial_metrics,
            "after": final_metrics
        }
        self.logger.metadata["execution"] = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "rows_per_second": len(self.data) / duration if duration > 0 else 0
        }
        
        self.logger.save_report(report_path)
        print("--- Pipeline Finished ---")


if __name__ == "__main__":
    # Example usage
    # preparer = AutoDataPreparer('data.csv')
    # preparer.run_pipeline()
    print("AutoDataPreparer is ready. Import and usage:")
    print("preparer = AutoDataPreparer('your_data.csv', target_col='target')")
    print("preparer.run_pipeline()")

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, mean_squared_error, r2_score
import os
import json
from datetime import datetime
from meta_learning import MetaLearner, MetaFeatureExtractor

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
    def __init__(self, file_path, target_col=None):
        self.file_path = file_path
        self.data = None
        self.raw_data_snapshot = None # Keep a copy for evaluation
        self.logger = DecisionLogger()
        self.target_col = target_col
        self.task_type = "classification" # Default 
        self.start_time = None
        
        # Initialize Meta-Learner
        self.meta_learner = MetaLearner()
        self.meta_learner.load_models()
        
    def _detect_task_type(self):
        """
        Detects if the problem is Classification or Regression based on target column.
        """
        if not self.target_col or self.data is None or self.target_col not in self.data.columns:
            return

        target_series = self.data[self.target_col]
        
        # Heuristic:
        # 1. If non-numeric -> Classification
        # 2. If numeric but low cardinality (<20 unique or <5% unique) -> Classification
        # 3. Else -> Regression
        
        if not pd.api.types.is_numeric_dtype(target_series):
            self.task_type = "classification"
        else:
            unique_count = target_series.nunique()
            total_count = len(target_series)
            if unique_count < 20 or (unique_count / total_count < 0.05):
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        
        self.logger.log("Task Detection", self.task_type.title(), f"Detected based on target '{self.target_col}' properties",
                       {"target_dtype": str(target_series.dtype), "unique_values": int(target_series.nunique())})

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

    def clean_data(self, drop_duplicates=True):
        """Cleans data by handling duplicates and outliers dynamically."""
        if self.data is None: return
        
        # Remove duplicates
        if drop_duplicates:
            initial_rows = len(self.data)
            self.data.drop_duplicates(inplace=True)
            dropped = initial_rows - len(self.data)
            if dropped > 0:
                self.logger.log("Deduplication", "Removed Duplicates", "Exact duplicate rows found", {"count": dropped})
            else:
                self.logger.log("Deduplication", "Skipped", "No duplicates found")

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

    def _evaluate_candidate_strategy(self, task, strategy, X, y):
        """
        Evaluates a specific strategy using a downstream model.
        Returns score (F1 for classification, R2 for regression).
        """
        try:
            # Build a temporary pipeline for evaluation
            steps = []
            
            # 1. Imputation
            if task == "imputation":
                if strategy == "knn": imputer = KNNImputer(n_neighbors=5)
                elif strategy == "median": imputer = SimpleImputer(strategy='median')
                elif strategy == "mean": imputer = SimpleImputer(strategy='mean')
                steps.append(('imputer', imputer))
            else:
                # Default safe imputer if we are testing something else
                steps.append(('imputer', SimpleImputer(strategy='mean'))) # Or median
            
            # 2. Encoding (For X only)
            # We need to handle categorical columns. 
            # If X is raw, we need an encoder.
            # Simplified: Use OneHot for evaluation to be safe.
            
            if task == "scaling":
                if strategy == "robust": scaler = RobustScaler()
                elif strategy == "minmax": scaler = MinMaxScaler()
                elif strategy == "standard": scaler = StandardScaler()
                steps.append(('scaler', scaler))
            else:
                steps.append(('scaler', StandardScaler()))
                
            # 3. Model & Metric Selection
            if self.task_type == "regression":
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                scoring = 'r2'
            else:
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                scoring = 'f1_macro'

            steps.append(('model', model))
            
            pipeline = Pipeline(steps)
            
            # Cross-validate
            # Ensure y is not null
            valid_mask = ~pd.isnull(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            scores = cross_val_score(pipeline, X_valid, y_valid, cv=3, scoring=scoring)
            return scores.mean()
        except Exception as e:
            # print(f"Eval failed for {strategy}: {e}")
            return -1e9 if self.task_type == "regression" else -1

    def handle_missing_values(self):
        """Handles missing values using Meta-Learning with Top-K Optimization."""
        if self.data is None: return
        
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) > 0:
            cols_with_missing = [c for c in num_cols if self.data[c].isnull().sum() > 0]
            
            if cols_with_missing:
                worst_col = self.data[cols_with_missing].isnull().sum().idxmax()
                meta_features = MetaFeatureExtractor.extract_features(self.data, worst_col)
                
                # Decision Logic
                if self.target_col and self.target_col in self.data.columns:
                    # OPTIMIZATION MODE
                    candidates = self.meta_learner.predict_top_k("imputation", meta_features, k=2)
                    self.logger.log("Imputation", "Optimizing...", f"Evaluating Top-K: {candidates}")
                    
                    best_score = -1
                    best_strategy = candidates[0]
                    
                    # Prepare X, y for evaluation
                    # We use numeric columns only for this test to avoid encoding issues
                    X_eval = self.data[num_cols].copy()
                    y_eval = self.data[self.target_col]
                    # Drop target from X if present
                    if self.target_col in X_eval.columns:
                        X_eval = X_eval.drop(columns=[self.target_col])

                    for strategy in candidates:
                        score = self._evaluate_candidate_strategy("imputation", strategy, X_eval, y_eval)
                        metric_name = "R2" if self.task_type == "regression" else "F1"
                        self.logger.log("Evaluation", f"{strategy}", f"{metric_name} Score: {score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                    
                    strategy = best_strategy
                    self.logger.log("Imputation", f"{strategy.title()} (Winner)", f"Selected after evaluation (Score: {best_score:.4f})", 
                                  {"candidates": candidates, "best_score": best_score})
                    
                    # Feedback Loop
                    self.meta_learner.store_feedback("imputation", meta_features, best_strategy)
                    
                else:
                    # FAST MODE (Prediction only)
                    strategy = self.meta_learner.predict("imputation", meta_features)
                    self.logger.log("Imputation", f"{strategy.title()} Strategy", "Predicted by Meta-Model", 
                                  {"predicted_strategy": strategy})
                
                # Apply Winner
                missing_mask = self.data[num_cols].isnull()
                for col in num_cols:
                    n_missing = missing_mask[col].sum()
                    if n_missing > 0:
                        self.logger.log_change("Imputation", col, "NaN", f"Filled using {strategy}", int(n_missing))

                if strategy == "knn":
                    imputer = KNNImputer(n_neighbors=5)
                    self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
                elif strategy == "median":
                    imputer = SimpleImputer(strategy='median')
                    self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
                elif strategy == "mean":
                    imputer = SimpleImputer(strategy='mean')
                    self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
        
        # Categorical Imputation
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                n_missing = self.data[col].isnull().sum()
                if n_missing > 0:
                    self.logger.log_change("Imputation", col, "NaN", "Most Frequent Value", int(n_missing))
            
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.data[cat_cols] = cat_imputer.fit_transform(self.data[cat_cols])
            self.logger.log("Imputation", "Most Frequent", "Categorical columns (Default)", {"columns": list(cat_cols)})

    def parse_dates(self):
        """Attempts to parse date columns safely."""
        if self.data is None: return
        
        for col in self.data.columns:
            # Only check object columns that are NOT already numeric
            if self.data[col].dtype == 'object':
                # Safe Check: Sample the column to see if it looks like a date
                non_na_sample = self.data[col].dropna().head(100)
                if len(non_na_sample) == 0: continue
                
                try:
                    # strict=False allows parsing mixed formats, but we want to be sure
                    # We check if >80% of non-null values can be parsed
                    parsed = pd.to_datetime(non_na_sample, errors='coerce')
                    success_rate = parsed.notna().mean()
                    
                    if success_rate > 0.8:
                        self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                        
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

        # Identify likely integer columns (IDs, Age, Year, Month, Day, Rating)
        # Heuristic: if column name suggests integer or if all non-null values are integers
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                col_lower = col.lower()
                is_int_candidate = False
                
                # Check 1: Name-based heuristic
                if any(x in col_lower for x in ['id', 'age', 'year', 'month', 'day', 'count', 'num']):
                    is_int_candidate = True
                
                # Check 2: Value-based heuristic (if almost all values are close to integers)
                if not is_int_candidate:
                     non_na = self.data[col].dropna()
                     if len(non_na) > 0:
                         is_close_to_int = np.allclose(non_na, np.round(non_na), atol=0.01)
                         if is_close_to_int:
                             is_int_candidate = True

                if is_int_candidate:
                    self.data[col] = self.data[col].round()
                    try:
                        self.data[col] = self.data[col].astype('Int64')
                        # self.logger.log("Type Enforcement", "Converted to Integer", f"Column {col} looks like integer", {"column": col})
                    except:
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

                # Target Column Handling: Always Label Encode for Classification
                if self.target_col and col == self.target_col:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.logger.log("Encoding", "Target Encoding", f"Label Encoded Target Variable", {"column": col})
                    continue

                # Meta-Learning Decision
                meta_features = MetaFeatureExtractor.extract_features(self.data, col)
                strategy = self.meta_learner.predict("encoding", meta_features)
                
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
            if self.target_col and self.target_col in self.data.columns:
                # OPTIMIZATION MODE
                candidates = self.meta_learner.predict_top_k("scaling", meta_features, k=2)
                self.logger.log("Scaling", "Optimizing...", f"Evaluating Top-K: {candidates}")
                
                best_score = -1
                best_strategy = candidates[0]
                
                # Prepare X, y for evaluation
                # Data is already encoded and numeric now
                X_eval = self.data[cols_to_scale].copy()
                y_eval = self.data[self.target_col]
                
                # Note: X_eval here only contains columns to scale. 
                # Ideally we should use ALL features for training to get a real score, 
                # but we only want to compare scalers on these columns.
                # Adding unscaled binary columns back to X_eval might be better for model performance.
                other_cols = [c for c in self.data.columns if c not in cols_to_scale and c != self.target_col]
                if other_cols:
                    X_eval = pd.concat([X_eval, self.data[other_cols]], axis=1)
                
                for strategy in candidates:
                    score = self._evaluate_candidate_strategy("scaling", strategy, X_eval, y_eval)
                    metric_name = "R2" if self.task_type == "regression" else "F1"
                    self.logger.log("Evaluation", f"{strategy}", f"{metric_name} Score: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
                
                strategy = best_strategy
                self.logger.log("Scaling", f"{strategy.title()} (Winner)", f"Selected after evaluation (Score: {best_score:.4f})", 
                              {"candidates": candidates, "best_score": best_score})
                
                # Feedback Loop
                self.meta_learner.store_feedback("scaling", meta_features, best_strategy)
                
            else:
                strategy = self.meta_learner.predict("scaling", meta_features)
            
            # Apply Winner
            if strategy == "robust":
                scaler = RobustScaler()
                if not self.target_col: self.logger.log("Scaling", "RobustScaler", "Predicted by Meta-Model")
            elif strategy == "minmax":
                scaler = MinMaxScaler()
                if not self.target_col: self.logger.log("Scaling", "MinMaxScaler", "Predicted by Meta-Model")
            else:
                scaler = StandardScaler()
                if not self.target_col: self.logger.log("Scaling", "StandardScaler", "Predicted by Meta-Model")
                
            for col in cols_to_scale:
                old_min, old_max = self.data[col].min(), self.data[col].max()
                self.logger.log_change("Scaling", col, f"Range [{old_min:.2f}, {old_max:.2f}]", f"Scaled using {strategy}", len(self.data))
                
            self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])

    def save_data(self, output_path):
        """Saves the processed data."""
        if self.data is None: return
        self.data.to_csv(output_path, index=False)
        # print(f"\nProcessed data saved to {output_path}")

    def run_pipeline(self, output_path='processed_data.csv'):
        """Runs the full preparation pipeline."""
        self.start_time = datetime.now()
        print("--- Pipeline Started ---")
        self.load_data()
        
        # Initial Evaluation
        initial_metrics = DataEvaluator.calculate_quality_metrics(self.data)
        
        self.analyze_data()
        self._detect_task_type() # Detect Classification vs Regression
        self.clean_data() # Dynamic Outlier Handling
        self.parse_dates() # Safer Date Parsing
        
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
        self.save_data(human_readable_path)
        self.logger.log("Output", "Save Human Readable", "Saved intermediate cleaned data", {"path": human_readable_path})

        self.encode_categorical() # Dynamic Thresholds
        self.scale_features() # Dynamic Scaler Selection
        self.analyze_data() 
        self.save_data(output_path)
        self.logger.log("Output", "Save ML Ready", "Saved final processed data", {"path": output_path})
        
        # Final Evaluation
        final_metrics = DataEvaluator.calculate_quality_metrics(self.data)
        
        # Save Report
        report_path = os.path.join(dirname, f'report_{basename.replace(".csv", ".json")}')
        
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

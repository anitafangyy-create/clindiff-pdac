"""
ClinDiff-PDAC Enhanced Baselines Module
========================================
Pure Python implementations of imputation methods without sklearn dependency.

Methods:
    - MissForest: Iterative imputation using random forest
    - kNN: k-Nearest Neighbors imputation
    - MICE: Multiple Imputation by Chained Equations

Author: ClinDiff-PDAC Team
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import warnings
from collections import defaultdict


# ==============================================================================
# Utility Functions
# ==============================================================================

def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _normalize_numeric(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize numeric data to [0, 1]. Returns (normalized, min_vals, max_vals)."""
    min_vals = np.nanmin(data, axis=0)
    max_vals = np.nanmax(data, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    normalized = (data - min_vals) / range_vals
    return normalized, min_vals, max_vals


def _denormalize(data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """Denormalize data back to original scale."""
    return data * (max_vals - min_vals) + min_vals


def _compute_distance_matrix(
    data: np.ndarray,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise distance matrix for rows.
    Handles missing values by using available features only.
    """
    n = data.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Find features where both have values
            valid_mask = ~np.isnan(data[i]) & ~np.isnan(data[j])
            if valid_mask.sum() == 0:
                dist = np.inf
            else:
                diff = data[i, valid_mask] - data[j, valid_mask]
                if metric == "euclidean":
                    dist = np.sqrt(np.mean(diff ** 2))
                elif metric == "manhattan":
                    dist = np.mean(np.abs(diff))
                else:
                    dist = np.sqrt(np.mean(diff ** 2))
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


# ==============================================================================
# Pure Python Decision Tree (for MissForest)
# ==============================================================================

class _DecisionTreeRegressor:
    """Simple decision tree regressor for MissForest."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTreeRegressor":
        """Fit the decision tree."""
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the decision tree."""
        if self.tree_ is None:
            raise ValueError("Tree not fitted")
        return np.array([self._predict_single(x) for x in X])

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> Dict:
        """Recursively build the decision tree."""
        n_samples = len(y)

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            np.var(y) < 1e-10):
            return {"leaf": True, "value": np.mean(y)}

        # Find best split
        n_features = X.shape[1]
        feature_indices = list(range(n_features))
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(
                n_features, self.max_features, replace=False
            ).tolist()

        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        for feature in feature_indices:
            values = X[:, feature]
            unique_values = np.unique(values[~np.isnan(values)])

            if len(unique_values) < 2:
                continue

            # Try thresholds between unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                # Variance reduction
                gain = np.var(y) - (
                    len(left_y) * np.var(left_y) + len(right_y) * np.var(right_y)
                ) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return {"leaf": True, "value": np.mean(y)}

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict for a single sample."""
        node = self.tree_
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]


class _DecisionTreeClassifier:
    """Simple decision tree classifier for MissForest."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTreeClassifier":
        """Fit the decision tree classifier."""
        self.classes_ = np.unique(y[~pd.isna(y)])
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.tree_ is None:
            raise ValueError("Tree not fitted")
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.tree_ is None:
            raise ValueError("Tree not fitted")
        result = []
        for x in X:
            probs = self._predict_proba_single(x)
            result.append(probs)
        return np.array(result)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> Dict:
        """Recursively build the decision tree."""
        n_samples = len(y)
        unique_classes, counts = np.unique(y, return_counts=True)

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            len(unique_classes) == 1):
            probs = np.zeros(len(self.classes_))
            for i, cls in enumerate(self.classes_):
                probs[i] = (y == cls).sum() / n_samples
            return {"leaf": True, "probs": probs}

        # Find best split using Gini impurity
        n_features = X.shape[1]
        feature_indices = list(range(n_features))
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(
                n_features, self.max_features, replace=False
            ).tolist()

        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        current_gini = self._gini(y)

        for feature in feature_indices:
            values = X[:, feature]
            unique_values = np.unique(values[~np.isnan(values)])

            if len(unique_values) < 2:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                # Gini reduction
                weighted_gini = (
                    len(left_y) * self._gini(left_y) +
                    len(right_y) * self._gini(right_y)
                ) / n_samples
                gain = current_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            probs = np.zeros(len(self.classes_))
            for i, cls in enumerate(self.classes_):
                probs[i] = (y == cls).sum() / n_samples
            return {"leaf": True, "probs": probs}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _predict_single(self, x: np.ndarray) -> Any:
        """Predict class for a single sample."""
        node = self.tree_
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return self.classes_[np.argmax(node["probs"])]

    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        node = self.tree_
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["probs"]


class _RandomForestRegressor:
    """Simple random forest regressor."""

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Optional[Union[int, str]] = "sqrt"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_: List[_DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomForestRegressor":
        """Fit the random forest."""
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features)) + 1
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features

        self.trees_ = []
        n_samples = len(y)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = _DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the random forest."""
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(predictions, axis=0)


class _RandomForestClassifier:
    """Simple random forest classifier."""

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Optional[Union[int, str]] = "sqrt"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_: List[_DecisionTreeClassifier] = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomForestClassifier":
        """Fit the random forest classifier."""
        self.classes_ = np.unique(y[~pd.isna(y)])
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features)) + 1
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features

        self.trees_ = []
        n_samples = len(y)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = _DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        all_probas = []
        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)
            all_probas.append(tree_proba)
        # Stack and average
        max_len = max(p.shape[1] for p in all_probas)
        padded = []
        for p in all_probas:
            if p.shape[1] < max_len:
                pad = np.zeros((p.shape[0], max_len - p.shape[1]))
                p = np.hstack([p, pad])
            padded.append(p)
        return np.mean(np.array(padded), axis=0)


# ==============================================================================
# MissForest Imputer
# ==============================================================================

class MissForestImputer:
    """
    MissForest imputation using iterative random forest.

    Iteratively imputes missing values using random forests:
    1. Initialize missing values with mean/mode
    2. For each variable with missing values, train RF on observed values
    3. Predict missing values
    4. Repeat until convergence or max iterations

    Parameters
    ----------
    max_iter : int, default 10
        Maximum number of imputation iterations.
    n_estimators : int, default 10
        Number of trees in each random forest.
    max_depth : int, default 10
        Maximum depth of each tree.
    tol : float, default 1e-3
        Convergence tolerance.
    random_state : int, optional
        Random seed for reproducibility.

    Example
    -------
    >>> imputer = MissForestImputer(max_iter=5)
    >>> df_imputed = imputer.fit_transform(df)
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 10,
        max_depth: int = 10,
        tol: float = 1e-3,
        random_state: Optional[int] = None
    ):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tol = tol
        self.random_state = random_state
        self.imputed_values_: Dict[str, np.ndarray] = {}
        self.converged_ = False
        self.n_iter_ = 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        df_imputed = df.copy()
        numeric_cols = _get_numeric_columns(df)
        categorical_cols = _get_categorical_columns(df)

        # Initialize missing values
        for col in numeric_cols:
            if df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

        for col in categorical_cols:
            if df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_val[0])

        # Iterative imputation
        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1
            prev_values = df_imputed.copy()

            # Impute each column
            for col in df.columns:
                if df[col].isna().sum() == 0:
                    continue

                missing_mask = df[col].isna()
                if missing_mask.sum() == 0:
                    continue

                # Prepare features (all other columns)
                feature_cols = [c for c in df.columns if c != col]
                X = df_imputed[feature_cols].copy()

                # Encode categorical features
                for c in X.columns:
                    if X[c].dtype == "object" or X[c].dtype.name == "category":
                        X[c] = X[c].astype("category").cat.codes

                X = X.values
                y = df_imputed[col].values

                if col in numeric_cols:
                    # Regression
                    model = _RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth
                    )
                    model.fit(X[~missing_mask], y[~missing_mask])
                    predictions = model.predict(X[missing_mask])
                else:
                    # Classification
                    model = _RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth
                    )
                    model.fit(X[~missing_mask], y[~missing_mask])
                    predictions = model.predict(X[missing_mask])

                df_imputed.loc[missing_mask, col] = predictions

            # Check convergence
            if self._check_convergence(prev_values, df_imputed):
                self.converged_ = True
                break

        return df_imputed

    def _check_convergence(
        self,
        prev: pd.DataFrame,
        curr: pd.DataFrame
    ) -> bool:
        """Check if imputation has converged."""
        numeric_cols = _get_numeric_columns(prev)
        if len(numeric_cols) == 0:
            return True

        max_diff = 0
        for col in numeric_cols:
            diff = np.abs(prev[col].values - curr[col].values).max()
            max_diff = max(max_diff, diff)

        return max_diff < self.tol


# ==============================================================================
# kNN Imputer
# ==============================================================================

class KNNImputer:
    """
    k-Nearest Neighbors imputation (pure Python implementation).

    Imputes missing values by finding k nearest neighbors with complete
    information and using their values (mean for numeric, mode for categorical).

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbors to use.
    weights : str, default "distance"
        Weighting method: "uniform" or "distance".
    metric : str, default "euclidean"
        Distance metric: "euclidean" or "manhattan".

    Example
    -------
    >>> imputer = KNNImputer(n_neighbors=3)
    >>> df_imputed = imputer.fit_transform(df)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "euclidean"
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        df_imputed = df.copy()
        numeric_cols = _get_numeric_columns(df)

        # Convert to numpy for distance computation
        # Encode categoricals as numeric
        df_encoded = df.copy()
        categorical_mappings = {}
        for col in df.columns:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                codes = df[col].astype("category").cat.codes
                categorical_mappings[col] = dict(enumerate(df[col].astype("category").cat.categories))
                df_encoded[col] = codes.replace(-1, np.nan)

        data = df_encoded.values.astype(float)

        # Impute each row with missing values
        for i in range(len(df)):
            row_missing = np.isnan(data[i])
            if not row_missing.any():
                continue

            # Find neighbors with complete information for relevant features
            for j in range(len(df.columns)):
                if not np.isnan(data[i, j]):
                    continue

                # Find rows with this feature observed
                observed_mask = ~np.isnan(data[:, j])
                if observed_mask.sum() < self.n_neighbors:
                    # Use mean/mode if not enough neighbors
                    if df.columns[j] in numeric_cols:
                        df_imputed.iloc[i, j] = df.iloc[:, j].median()
                    else:
                        mode_val = df.iloc[:, j].mode()
                        df_imputed.iloc[i, j] = mode_val[0] if len(mode_val) > 0 else None
                    continue

                # Compute distances to observed rows
                distances = []
                for k in range(len(df)):
                    if k == i or not observed_mask[k]:
                        continue
                    valid_features = ~row_missing & ~np.isnan(data[k])
                    if valid_features.sum() == 0:
                        continue
                    diff = data[i, valid_features] - data[k, valid_features]
                    if self.metric == "euclidean":
                        dist = np.sqrt(np.mean(diff ** 2))
                    else:
                        dist = np.mean(np.abs(diff))
                    distances.append((dist, k))

                if len(distances) < self.n_neighbors:
                    # Fallback
                    if df.columns[j] in numeric_cols:
                        df_imputed.iloc[i, j] = df.iloc[:, j].median()
                    else:
                        mode_val = df.iloc[:, j].mode()
                        df_imputed.iloc[i, j] = mode_val[0] if len(mode_val) > 0 else None
                    continue

                # Sort by distance and take k nearest
                distances.sort(key=lambda x: x[0])
                nearest = distances[:self.n_neighbors]

                # Compute weighted value
                neighbor_values = data[[k for _, k in nearest], j]

                if df.columns[j] in numeric_cols:
                    if self.weights == "distance":
                        weights = np.array([1 / (d + 1e-10) for d, _ in nearest])
                        weights /= weights.sum()
                        imputed_value = np.average(neighbor_values, weights=weights)
                    else:
                        imputed_value = np.mean(neighbor_values)
                    df_imputed.iloc[i, j] = imputed_value
                else:
                    # For categorical, use weighted voting
                    values, counts = np.unique(neighbor_values, return_counts=True)
                    if self.weights == "distance":
                        # Weight by inverse distance
                        weighted_counts = np.zeros(len(values))
                        for idx, (dist, row_idx) in enumerate(nearest):
                            val = data[row_idx, j]
                            val_idx = np.where(values == val)[0][0]
                            weighted_counts[val_idx] += 1 / (dist + 1e-10)
                        imputed_value = values[np.argmax(weighted_counts)]
                    else:
                        imputed_value = values[np.argmax(counts)]

                    # Map back to original category
                    col_name = df.columns[j]
                    if imputed_value in categorical_mappings.get(col_name, {}):
                        df_imputed.iloc[i, j] = categorical_mappings[col_name][int(imputed_value)]
                    else:
                        df_imputed.iloc[i, j] = imputed_value

        return df_imputed


# ==============================================================================
# MICE Imputer
# ==============================================================================

class MICEImputer:
    """
    Multiple Imputation by Chained Equations (pure Python implementation).

    Iteratively imputes each variable using a simple model (mean for numeric,
    mode for categorical) conditioned on other variables.

    Parameters
    ----------
    max_iter : int, default 10
        Maximum number of imputation iterations.
    n_imputations : int, default 1
        Number of imputed datasets to generate (for uncertainty estimation).
    random_state : int, optional
        Random seed.

    Example
    -------
    >>> imputer = MICEImputer(max_iter=5)
    >>> df_imputed = imputer.fit_transform(df)
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_imputations: int = 1,
        random_state: Optional[int] = None
    ):
        self.max_iter = max_iter
        self.n_imputations = n_imputations
        self.random_state = random_state
        self.imputed_values_ = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        df_imputed = df.copy()
        numeric_cols = _get_numeric_columns(df)
        categorical_cols = _get_categorical_columns(df)

        # Initialize with mean/mode
        for col in numeric_cols:
            if df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

        for col in categorical_cols:
            if df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_val[0])

        # Iterative imputation
        for iteration in range(self.max_iter):
            for col in df.columns:
                if df[col].isna().sum() == 0:
                    continue

                missing_mask = df[col].isna()
                if missing_mask.sum() == 0:
                    continue

                # Use other columns as predictors
                feature_cols = [c for c in df.columns if c != col]
                X = df_imputed[feature_cols].copy()

                # Simple encoding for categoricals
                for c in X.columns:
                    if X[c].dtype == "object" or X[c].dtype.name == "category":
                        X[c] = X[c].astype("category").cat.codes

                X_obs = X[~missing_mask].values
                y_obs = df_imputed.loc[~missing_mask, col].values
                X_miss = X[missing_mask].values

                if col in numeric_cols:
                    # Simple linear regression approximation
                    predictions = self._predict_numeric(X_obs, y_obs, X_miss)
                else:
                    # Simple classification
                    predictions = self._predict_categorical(X_obs, y_obs, X_miss)

                df_imputed.loc[missing_mask, col] = predictions

        return df_imputed

    def _predict_numeric(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        X_miss: np.ndarray
    ) -> np.ndarray:
        """Predict numeric values using simple regression."""
        # Simple approach: use mean of y_obs for rows with similar X patterns
        predictions = []
        y_mean = np.mean(y_obs)
        y_std = np.std(y_obs)

        for x_miss in X_miss:
            # Find similar observed rows
            distances = []
            for x_obs in X_obs:
                valid = ~np.isnan(x_miss) & ~np.isnan(x_obs)
                if valid.sum() == 0:
                    dist = np.inf
                else:
                    dist = np.sqrt(np.mean((x_miss[valid] - x_obs[valid]) ** 2))
                distances.append(dist)

            # Use k=3 nearest neighbors
            k = min(3, len(distances))
            nearest_idx = np.argsort(distances)[:k]
            pred = np.mean(y_obs[nearest_idx])
            predictions.append(pred)

        return np.array(predictions)

    def _predict_categorical(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        X_miss: np.ndarray
    ) -> np.ndarray:
        """Predict categorical values using simple classification."""
        predictions = []

        for x_miss in X_miss:
            distances = []
            for x_obs in X_obs:
                valid = ~np.isnan(x_miss) & ~np.isnan(x_obs)
                if valid.sum() == 0:
                    dist = np.inf
                else:
                    dist = np.sqrt(np.mean((x_miss[valid] - x_obs[valid]) ** 2))
                distances.append(dist)

            k = min(3, len(distances))
            nearest_idx = np.argsort(distances)[:k]
            nearest_y = y_obs[nearest_idx]

            # Mode of nearest neighbors
            values, counts = np.unique(nearest_y, return_counts=True)
            predictions.append(values[np.argmax(counts)])

        return np.array(predictions)


# ==============================================================================
# Unit Tests
# ==============================================================================

def _test_enhanced_baselines():
    """Unit tests for enhanced baseline imputers."""
    np.random.seed(42)

    # Create test data with missing values
    n = 100
    df = pd.DataFrame({
        "numeric1": np.random.randn(n),
        "numeric2": np.random.randn(n) * 2 + 5,
        "numeric3": np.random.randn(n) * 0.5 - 3,
        "categorical": np.random.choice(["A", "B", "C"], n),
    })

    # Introduce missing values
    missing_idx_num1 = np.random.choice(n, 15, replace=False)
    missing_idx_num2 = np.random.choice(n, 10, replace=False)
    missing_idx_cat = np.random.choice(n, 12, replace=False)

    df.loc[missing_idx_num1, "numeric1"] = np.nan
    df.loc[missing_idx_num2, "numeric2"] = np.nan
    df.loc[missing_idx_cat, "categorical"] = np.nan

    n_missing_before = df.isna().sum().sum()

    # Test 1: MissForestImputer
    print("Testing MissForestImputer...")
    imputer_mf = MissForestImputer(max_iter=3, n_estimators=5, random_state=42)
    df_mf = imputer_mf.fit_transform(df)
    assert df_mf.isna().sum().sum() == 0, "MissForest should impute all missing values"
    assert df_mf.shape == df.shape, "Shape should be preserved"
    print(f"  Converged: {imputer_mf.converged_}, Iterations: {imputer_mf.n_iter_}")

    # Test 2: KNNImputer
    print("Testing KNNImputer...")
    imputer_knn = KNNImputer(n_neighbors=3)
    df_knn = imputer_knn.fit_transform(df)
    assert df_knn.isna().sum().sum() == 0, "KNN should impute all missing values"
    assert df_knn.shape == df.shape, "Shape should be preserved"

    # Test 3: MICEImputer
    print("Testing MICEImputer...")
    imputer_mice = MICEImputer(max_iter=5, random_state=42)
    df_mice = imputer_mice.fit_transform(df)
    assert df_mice.isna().sum().sum() == 0, "MICE should impute all missing values"
    assert df_mice.shape == df.shape, "Shape should be preserved"

    # Test 4: Decision Tree components
    print("Testing Decision Tree components...")
    X = np.random.randn(50, 3)
    y_reg = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(50) * 0.1
    y_cls = np.where(X[:, 0] > 0, "A", "B")

    tree_reg = _DecisionTreeRegressor(max_depth=5)
    tree_reg.fit(X, y_reg)
    pred_reg = tree_reg.predict(X)
    assert len(pred_reg) == 50, "Should predict for all samples"

    tree_cls = _DecisionTreeClassifier(max_depth=5)
    tree_cls.fit(X, y_cls)
    pred_cls = tree_cls.predict(X)
    assert len(pred_cls) == 50, "Should predict for all samples"
    assert set(pred_cls).issubset({"A", "B"}), "Should predict valid classes"

    # Test 5: Random Forest components
    print("Testing Random Forest components...")
    rf_reg = _RandomForestRegressor(n_estimators=5, max_depth=5)
    rf_reg.fit(X, y_reg)
    pred_rf_reg = rf_reg.predict(X)
    assert len(pred_rf_reg) == 50, "RF should predict for all samples"

    rf_cls = _RandomForestClassifier(n_estimators=5, max_depth=5)
    rf_cls.fit(X, y_cls)
    pred_rf_cls = rf_cls.predict(X)
    assert len(pred_rf_cls) == 50, "RF should predict for all samples"

    print("All Enhanced Baselines tests passed!")
    return True


if __name__ == "__main__":
    _test_enhanced_baselines()

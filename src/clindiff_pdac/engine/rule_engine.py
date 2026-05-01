"""
ClinDiff-PDAC Rule Engine Module
=================================
Four-state missingness mask generation for clinical EMR data.

States:
    1 = Observed: Value exists and is valid
    0 = True Missing: No value but theoretically should exist
   NA = Not Applicable: Field does not apply to this patient
  OTW = Out of Time Window: Value exists but outside allowed time range

Author: ClinDiff-PDAC Team
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import IntEnum
import warnings


class MaskState(IntEnum):
    """Four-state mask values for missingness classification."""
    OBSERVED = 1
    TRUE_MISSING = 0
    NOT_APPLICABLE = -1
    OUT_OF_TIME_WINDOW = -2


@dataclass
class TimeWindow:
    """Time window specification for a clinical variable."""
    lower_bound_days: float  # Days relative to anchor (e.g., diagnosis date)
    upper_bound_days: float
    anchor_field: str = "diagnosis_date"

    def contains(self, value_days: float) -> bool:
        """Check if a time point falls within the window."""
        if pd.isna(value_days):
            return False
        return self.lower_bound_days <= value_days <= self.upper_bound_days


@dataclass
class ApplicabilityRule:
    """Rule defining when a clinical variable applies to a patient."""
    condition_column: str
    condition_operator: str  # "==", "!=", "in", "not in", "is_null", "not_null"
    condition_value: Any


@dataclass
class VariableSpec:
    """Specification for a clinical variable's missingness rules."""
    column: str
    data_type: str = "numeric"  # "numeric", "categorical", "datetime"
    time_window: Optional[TimeWindow] = None
    applicability_rules: List[ApplicabilityRule] = field(default_factory=list)
    valid_range: Optional[Tuple[float, float]] = None  # For numeric
    valid_categories: Optional[List[Any]] = None  # For categorical
    na_values: List[Any] = field(default_factory=lambda: ["", "NA", "N/A", "null", "NULL", "None", "."])


class RuleEngine:
    """
    Four-state missingness mask generator for clinical EMR data.

    Generates a mask DataFrame with four possible states per cell:
    - 1: Observed (valid value exists)
    - 0: True Missing (no value but applicable)
    - -1: Not Applicable (not applicable to this patient)
    - -2: Out of Time Window (value exists but outside time range)

    Example
    -------
    >>> spec = VariableSpec("CA19_9", time_window=TimeWindow(-30, 7))
    >>> engine = RuleEngine({"CA19_9": spec})
    >>> mask = engine.generate_mask(df, patient_ids=["P001"], anchor_dates={"P001": pd.Timestamp("2024-01-01")})
    """

    def __init__(
        self,
        variable_specs: Optional[Dict[str, VariableSpec]] = None,
        default_time_window: Optional[TimeWindow] = None,
        default_na_values: Optional[List[str]] = None
    ):
        """
        Initialize the rule engine.

        Parameters
        ----------
        variable_specs : dict, optional
            Mapping from column name to VariableSpec.
        default_time_window : TimeWindow, optional
            Default time window for variables without explicit specs.
        default_na_values : list, optional
            Default values treated as missing (e.g., "", "NA").
        """
        self.variable_specs = variable_specs or {}
        self.default_time_window = default_time_window
        self.default_na_values = default_na_values or ["", "NA", "N/A", "null", "NULL", "None", "."]
        self._compiled_rules: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_mask(
        self,
        df: pd.DataFrame,
        patient_ids: Optional[List[str]] = None,
        anchor_dates: Optional[Dict[str, pd.Timestamp]] = None,
        date_columns: Optional[Dict[str, str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Generate four-state missingness mask for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input clinical data.
        patient_ids : list of str, optional
            List of patient IDs to process. If None, processes all rows.
        anchor_dates : dict, optional
            Mapping from patient_id to anchor date (e.g., diagnosis date).
            Required when any variable has a time window defined.
        date_columns : dict, optional
            Mapping from variable name to its associated date column name.
            E.g., {"CA19_9": "CA19_9_date", "CEA": "CEA_date"}
        inplace : bool, default False
            If True, modify the input DataFrame (adds 'mask' column).

        Returns
        -------
        pd.DataFrame
            DataFrame with mask columns for each variable. Each cell contains
            a MaskState integer value (1, 0, -1, -2).
        """
        if patient_ids is not None:
            mask_df = df[df.index.isin(patient_ids) if df.index.name else df.index.astype(str).isin(patient_ids)].copy()
        else:
            mask_df = df.copy()

        result = mask_df.copy()
        columns_to_process = self._get_columns_to_process(mask_df)

        for col in columns_to_process:
            spec = self.variable_specs.get(col)
            date_col = date_columns.get(col) if date_columns else None
            mask_col = self._generate_column_mask(
                mask_df, col, spec, anchor_dates, date_col
            )
            result[f"{col}_mask"] = mask_col

        return result

    def generate_mask_array(
        self,
        df: pd.DataFrame,
        patient_ids: Optional[List[str]] = None,
        anchor_dates: Optional[Dict[str, pd.Timestamp]] = None,
        date_columns: Optional[Dict[str, str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate masks as numpy arrays (useful for ML pipelines).

        Returns
        -------
        dict
            Mapping from column name to mask numpy array.
        """
        mask_df = self.generate_mask(df, patient_ids, anchor_dates, date_columns)
        result = {}
        for col in mask_df.columns:
            if col.endswith("_mask"):
                result[col.replace("_mask", "")] = mask_df[col].values
        return result

    def get_missingness_summary(
        self,
        mask_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute summary statistics from a mask DataFrame.

        Parameters
        ----------
        mask_df : pd.DataFrame
            Output from generate_mask().

        Returns
        -------
        pd.DataFrame
            Summary with columns: variable, observed, true_missing,
            not_applicable, out_of_time_window, total.
        """
        rows = []
        for col in mask_df.columns:
            if not col.endswith("_mask"):
                continue
            var_name = col.replace("_mask", "")
            series = mask_df[col]
            n_total = len(series)
            n_observed = int((series == MaskState.OBSERVED).sum())
            n_true_missing = int((series == MaskState.TRUE_MISSING).sum())
            n_not_applicable = int((series == MaskState.NOT_APPLICABLE).sum())
            n_otw = int((series == MaskState.OUT_OF_TIME_WINDOW).sum())
            rows.append({
                "variable": var_name,
                "observed": n_observed,
                "true_missing": n_true_missing,
                "not_applicable": n_not_applicable,
                "out_of_time_window": n_otw,
                "total": n_total,
                "observed_pct": round(n_observed / n_total * 100, 2) if n_total > 0 else 0
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Internal Logic
    # ------------------------------------------------------------------ #

    def _get_columns_to_process(self, df: pd.DataFrame) -> List[str]:
        """Determine which columns to generate masks for."""
        if self.variable_specs:
            # Only process columns that exist in the DataFrame
            return [col for col in self.variable_specs.keys() if col in df.columns]
        # Default: all columns except known metadata
        exclude = {"patient_id", "id", "pid", "record_id", "index", "date", "datetime"}
        return [c for c in df.columns if c.lower() not in exclude]

    def _generate_column_mask(
        self,
        df: pd.DataFrame,
        col: str,
        spec: Optional[VariableSpec],
        anchor_dates: Optional[Dict[str, pd.Timestamp]],
        date_col: Optional[str]
    ) -> np.ndarray:
        """Generate mask array for a single column."""
        n = len(df)
        mask = np.full(n, MaskState.TRUE_MISSING, dtype=np.int8)

        # Step 1: Check applicability
        is_applicable = self._check_applicability(df, col, spec)

        # Step 2: Check if value exists (not NA)
        has_value = self._check_has_value(df[col], spec)

        # Step 3: Check time window
        if spec and spec.time_window and anchor_dates and date_col:
            time_ok = self._check_time_window(df, date_col, spec.time_window, anchor_dates)
        else:
            time_ok = np.ones(n, dtype=bool)

        # Step 4: Assign states
        # NA: not applicable
        mask[~is_applicable] = MaskState.NOT_APPLICABLE

        # OTW: applicable but value is outside time window
        applicable_and_has_value = is_applicable & has_value
        applicable_and_has_value_otw = applicable_and_has_value & ~time_ok
        mask[applicable_and_has_value_otw] = MaskState.OUT_OF_TIME_WINDOW

        # Observed: applicable, has value, within time window
        fully_observed = applicable_and_has_value & time_ok
        mask[fully_observed] = MaskState.OBSERVED

        # True Missing: applicable but no value
        applicable_no_value = is_applicable & ~has_value & time_ok
        mask[applicable_no_value] = MaskState.TRUE_MISSING

        return mask

    def _check_applicability(
        self,
        df: pd.DataFrame,
        col: str,
        spec: Optional[VariableSpec]
    ) -> np.ndarray:
        """Determine which rows the variable applies to."""
        n = len(df)
        if spec is None or not spec.applicability_rules:
            return np.ones(n, dtype=bool)

        result = np.ones(n, dtype=bool)
        for rule in spec.applicability_rules:
            if rule.condition_column not in df.columns:
                warnings.warn(f"Applicability rule references missing column: {rule.condition_column}")
                return np.zeros(n, dtype=bool)

            col_data = df[rule.condition_column].values
            condition_holds = self._evaluate_condition(col_data, rule)
            result &= condition_holds

        return result

    def _evaluate_condition(
        self,
        data: np.ndarray,
        rule: ApplicabilityRule
    ) -> np.ndarray:
        """Evaluate a single applicability condition."""
        op = rule.condition_operator
        val = rule.condition_value

        if op == "==":
            return data == val
        elif op == "!=":
            return data != val
        elif op == "in":
            return np.isin(data, val if isinstance(val, (list, tuple, set, np.ndarray)) else [val])
        elif op == "not in":
            return ~np.isin(data, val if isinstance(val, (list, tuple, set, np.ndarray)) else [val])
        elif op == "is_null":
            return pd.isna(data)
        elif op == "not_null":
            return ~pd.isna(data)
        else:
            warnings.warn(f"Unknown operator: {op}")
            return np.zeros(len(data), dtype=bool)

    def _check_has_value(
        self,
        series: pd.Series,
        spec: Optional[VariableSpec]
    ) -> np.ndarray:
        """Check if each cell contains a valid (non-missing) value."""
        na_values = self.default_na_values
        if spec and spec.na_values:
            na_values = spec.na_values

        # Standard NA check
        not_na = ~series.isna()

        # Check against custom NA values
        if na_values:
            for na_val in na_values:
                not_na |= (series.astype(str).str.strip().str.upper() == str(na_val).upper())

        return not_na.values

    def _check_time_window(
        self,
        df: pd.DataFrame,
        date_col: str,
        time_window: TimeWindow,
        anchor_dates: Dict[str, pd.Timestamp]
    ) -> np.ndarray:
        """Check which values fall within the allowed time window."""
        n = len(df)
        result = np.zeros(n, dtype=bool)

        if date_col not in df.columns:
            warnings.warn(f"Date column '{date_col}' not found for time window check.")
            return result

        for i, (idx, row) in enumerate(df.iterrows()):
            patient_id = idx if df.index.name else str(idx)
            if patient_id not in anchor_dates:
                continue

            anchor = anchor_dates[patient_id]
            var_date = row[date_col]

            if pd.isna(var_date) or pd.isna(anchor):
                continue

            # Convert to days
            try:
                days_diff = (pd.Timestamp(var_date) - pd.Timestamp(anchor)).days
                result[i] = time_window.contains(days_diff)
            except Exception:
                continue

        return result

    def add_rule(
        self,
        column: str,
        spec: VariableSpec
    ) -> None:
        """Add or update a variable specification."""
        self.variable_specs[column] = spec

    def register_default_ca19_9_rules(self) -> None:
        """
        Register standard CA19-9 rules for PDAC.
        CA19-9 is Not Applicable for Lewis-a/b negative patients.
        """
        self.variable_specs["CA19_9"] = VariableSpec(
            column="CA19_9",
            data_type="numeric",
            time_window=TimeWindow(lower_bound_days=-30, upper_bound_days=7),
            applicability_rules=[
                ApplicabilityRule(
                    condition_column="Lewis_status",
                    condition_operator="not in",
                    condition_value=["negative", "a-b-", "Le(a-b-)"]
                )
            ],
            valid_range=(0, 1000000)
        )

    def register_default_pDAC_rules(self) -> None:
        """Register common PDAC clinical variable rules."""
        specs = [
            VariableSpec("CA19_9", time_window=TimeWindow(-30, 7)),
            VariableSpec("CEA", time_window=TimeWindow(-30, 7)),
            VariableSpec("ALB", time_window=TimeWindow(-7, 1)),
            VariableSpec("TBIL", time_window=TimeWindow(-7, 1)),
            VariableSpec("ALP", time_window=TimeWindow(-7, 1)),
            VariableSpec("T_stage", data_type="categorical"),
            VariableSpec("N_stage", data_type="categorical"),
            VariableSpec("M_stage", data_type="categorical"),
            VariableSpec("surgical_approach", data_type="categorical"),
            VariableSpec("resection_margin", data_type="categorical"),
        ]
        for spec in specs:
            self.variable_specs[spec.column] = spec


# ---------------------------------------------------------------------- #
# Unit Tests
# ---------------------------------------------------------------------- #

def _test_rule_engine():
    """Unit tests for RuleEngine."""
    import pandas as pd

    # Create test data with patient_id as index
    df = pd.DataFrame({
        "patient_id": ["P001", "P002", "P003", "P004"],
        "CA19_9": [125.0, np.nan, 340.0, np.nan],
        "CA19_9_date": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-15", "2024-01-20"]),
        "Lewis_status": ["positive", "negative", "positive", "negative"],
        "CEA": [5.2, 3.1, np.nan, 4.8],
        "diagnosis_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]),
    }).set_index("patient_id")

    anchor_dates = {
        "P001": pd.Timestamp("2024-01-01"),
        "P002": pd.Timestamp("2024-01-01"),
        "P003": pd.Timestamp("2024-01-01"),
        "P004": pd.Timestamp("2024-01-01"),
    }

    # Test 1: Basic rule engine
    engine = RuleEngine()
    mask_df = engine.generate_mask(df, anchor_dates=anchor_dates, date_columns={"CA19_9": "CA19_9_date"})
    assert "CA19_9_mask" in mask_df.columns, "CA19_9 mask column not generated"
    assert mask_df["CA19_9_mask"].dtype in [np.int8, np.int16, np.int32, np.int64], "Mask should be integer"

    # Test 2: Applicability rule (Lewis negative -> NA)
    engine2 = RuleEngine()
    engine2.register_default_ca19_9_rules()
    mask_df2 = engine2.generate_mask(df, anchor_dates=anchor_dates, date_columns={"CA19_9": "CA19_9_date"})
    # P002 and P004 are Lewis negative -> should be NA
    assert mask_df2.loc["P002", "CA19_9_mask"] == MaskState.NOT_APPLICABLE, "Lewis negative should be NA"
    assert mask_df2.loc["P004", "CA19_9_mask"] == MaskState.NOT_APPLICABLE, "Lewis negative should be NA"

    # Test 3: Time window
    engine3 = RuleEngine()
    spec = VariableSpec("CA19_9", time_window=TimeWindow(lower_bound_days=-30, upper_bound_days=7))
    engine3.add_rule("CA19_9", spec)
    mask_df3 = engine3.generate_mask(df, anchor_dates=anchor_dates, date_columns={"CA19_9": "CA19_9_date"})
    # P003 has CA19_9 on day 14, outside window -> OTW
    assert mask_df3.loc["P003", "CA19_9_mask"] == MaskState.OUT_OF_TIME_WINDOW, "Outside window should be OTW"
    # P001 has CA19_9 on day 4, inside window -> Observed
    assert mask_df3.loc["P001", "CA19_9_mask"] == MaskState.OBSERVED, "Within window should be Observed"

    # Test 4: Missingness summary
    summary = engine3.get_missingness_summary(mask_df3)
    assert len(summary) > 0, "Summary should not be empty"
    assert "observed" in summary.columns, "Summary should have observed column"
    assert "true_missing" in summary.columns, "Summary should have true_missing column"

    # Test 5: Observed + True Missing
    # P002 has Lewis negative -> NA (not applicable). No value = True Missing since Lewis negative makes it NA.
    # P001: day 4, value=125 -> Observed
    # P002: Lewis negative -> NA
    # P003: day 14 -> OTW (has value but outside window)
    # P004: Lewis negative -> NA
    print("All RuleEngine tests passed!")
    return True


if __name__ == "__main__":
    _test_rule_engine()

"""
ClinDiff-PDAC LLM Constraints Module
=====================================
LLM-enhanced imputation with structured constraints and evidence extraction.

Features:
    - Constrained completion for categorical fields
    - Structured output with confidence scores
    - Evidence extraction with explanations
    - Domain-aware prompting for clinical data

Author: ClinDiff-PDAC Team
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum
import warnings


class ConfidenceLevel(Enum):
    """Confidence levels for LLM predictions."""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5
    UNCERTAIN = "uncertain"


@dataclass
class ImputationResult:
    """Structured result from LLM imputation."""
    variable: str
    imputed_value: Any
    confidence_score: float  # 0-1
    confidence_level: ConfidenceLevel
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    alternative_values: List[Tuple[Any, float]] = field(default_factory=list)
    constraints_satisfied: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "imputed_value": self.imputed_value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "alternative_values": self.alternative_values,
            "constraints_satisfied": self.constraints_satisfied,
            "validation_errors": self.validation_errors
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class Constraint:
    """Constraint definition for a variable."""
    variable: str
    constraint_type: str  # "categorical", "range", "regex", "custom"
    allowed_values: Optional[List[Any]] = None
    value_range: Optional[Tuple[float, float]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    description: str = ""

    def validate(self, value: Any) -> Tuple[bool, List[str]]:
        """
        Validate a value against this constraint.

        Returns
        -------
        tuple
            (is_valid, list_of_errors)
        """
        errors = []

        if self.constraint_type == "categorical":
            if self.allowed_values is not None:
                if value not in self.allowed_values:
                    errors.append(
                        f"Value '{value}' not in allowed values: {self.allowed_values}"
                    )

        elif self.constraint_type == "range":
            if self.value_range is not None:
                try:
                    num_val = float(value)
                    if num_val < self.value_range[0] or num_val > self.value_range[1]:
                        errors.append(
                            f"Value {value} outside range [{self.value_range[0]}, {self.value_range[1]}]"
                        )
                except (ValueError, TypeError):
                    errors.append(f"Cannot convert '{value}' to numeric for range check")

        elif self.constraint_type == "regex":
            if self.pattern is not None:
                if not re.match(self.pattern, str(value)):
                    errors.append(f"Value '{value}' does not match pattern '{self.pattern}'")

        elif self.constraint_type == "custom":
            if self.custom_validator is not None:
                if not self.custom_validator(value):
                    errors.append(f"Value '{value}' failed custom validation")

        return len(errors) == 0, errors


@dataclass
class ClinicalContext:
    """Clinical context for LLM imputation."""
    patient_id: Optional[str] = None
    diagnosis: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    stage: Optional[str] = None
    comorbidities: List[str] = field(default_factory=list)
    prior_treatments: List[str] = field(default_factory=list)
    relevant_labs: Dict[str, float] = field(default_factory=dict)
    imaging_findings: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None  # e.g., "pre-operative", "post-chemo"

    def to_prompt_context(self) -> str:
        """Convert to prompt-friendly string."""
        parts = []
        if self.diagnosis:
            parts.append(f"Diagnosis: {self.diagnosis}")
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.gender:
            parts.append(f"Gender: {self.gender}")
        if self.stage:
            parts.append(f"Stage: {self.stage}")
        if self.comorbidities:
            parts.append(f"Comorbidities: {', '.join(self.comorbidities)}")
        if self.prior_treatments:
            parts.append(f"Prior treatments: {', '.join(self.prior_treatments)}")
        if self.relevant_labs:
            labs_str = ", ".join([f"{k}={v}" for k, v in self.relevant_labs.items()])
            parts.append(f"Relevant labs: {labs_str}")
        if self.temporal_context:
            parts.append(f"Temporal context: {self.temporal_context}")
        return "\n".join(parts)


class LLMConstraintLayer:
    """
    LLM-enhanced imputation with structured constraints and evidence extraction.

    This class provides:
    1. Constrained completion for categorical and numeric fields
    2. Structured output with confidence scores
    3. Evidence extraction with clinical reasoning
    4. Domain-aware prompting for PDAC clinical data

    Parameters
    ----------
    constraints : dict, optional
        Mapping from variable name to Constraint object.
    default_confidence_threshold : float, default 0.7
        Minimum confidence threshold for accepting imputations.

    Example
    -------
    >>> constraints = {
    ...     "T_stage": Constraint("T_stage", "categorical", allowed_values=["T1", "T2", "T3", "T4"]),
    ...     "CA19_9": Constraint("CA19_9", "range", value_range=(0, 100000))
    ... }
    >>> llm_layer = LLMConstraintLayer(constraints)
    >>> result = llm_layer.impute("T_stage", patient_data, clinical_context)
    """

    def __init__(
        self,
        constraints: Optional[Dict[str, Constraint]] = None,
        default_confidence_threshold: float = 0.7
    ):
        """
        Initialize the LLM constraint layer.

        Parameters
        ----------
        constraints : dict, optional
            Mapping from variable name to Constraint.
        default_confidence_threshold : float, default 0.7
            Minimum confidence for accepting imputations.
        """
        self.constraints = constraints or {}
        self.default_confidence_threshold = default_confidence_threshold
        self._imputation_history: List[ImputationResult] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def impute(
        self,
        variable: str,
        patient_data: pd.Series,
        clinical_context: Optional[ClinicalContext] = None,
        constraint: Optional[Constraint] = None,
        return_alternatives: bool = False
    ) -> ImputationResult:
        """
        Impute a single missing value using LLM with constraints.

        Parameters
        ----------
        variable : str
            Name of the variable to impute.
        patient_data : pd.Series
            Available data for the patient.
        clinical_context : ClinicalContext, optional
            Additional clinical context.
        constraint : Constraint, optional
            Constraint for this variable (overrides registered constraint).
        return_alternatives : bool, default False
            Whether to return alternative values with probabilities.

        Returns
        -------
        ImputationResult
            Structured imputation result with confidence and evidence.
        """
        # Get constraint
        if constraint is None:
            constraint = self.constraints.get(variable)

        # Build prompt
        prompt = self._build_imputation_prompt(
            variable, patient_data, clinical_context, constraint
        )

        # TODO: Call LLM API here
        # For now, return a placeholder result
        llm_response = self._call_llm_placeholder(prompt, variable, constraint)

        # Parse and validate response
        result = self._parse_llm_response(
            llm_response, variable, constraint, return_alternatives
        )

        # Validate against constraint
        if constraint:
            is_valid, errors = constraint.validate(result.imputed_value)
            result.constraints_satisfied = is_valid
            result.validation_errors = errors

        # Store in history
        self._imputation_history.append(result)

        return result

    def impute_batch(
        self,
        variables: List[str],
        patient_data: pd.Series,
        clinical_context: Optional[ClinicalContext] = None
    ) -> Dict[str, ImputationResult]:
        """
        Impute multiple variables for a single patient.

        Parameters
        ----------
        variables : list
            List of variable names to impute.
        patient_data : pd.Series
            Available patient data.
        clinical_context : ClinicalContext, optional
            Clinical context.

        Returns
        -------
        dict
            Mapping from variable name to ImputationResult.
        """
        results = {}
        for var in variables:
            results[var] = self.impute(var, patient_data, clinical_context)
        return results

    def add_constraint(self, constraint: Constraint) -> None:
        """Register a constraint for a variable."""
        self.constraints[constraint.variable] = constraint

    def register_default_pdac_constraints(self) -> None:
        """Register standard PDAC clinical variable constraints."""
        default_constraints = [
            Constraint(
                variable="T_stage",
                constraint_type="categorical",
                allowed_values=["T1", "T2", "T3", "T4", "Tx"],
                description="Tumor stage (TNM)"
            ),
            Constraint(
                variable="N_stage",
                constraint_type="categorical",
                allowed_values=["N0", "N1", "N2", "Nx"],
                description="Nodal stage (TNM)"
            ),
            Constraint(
                variable="M_stage",
                constraint_type="categorical",
                allowed_values=["M0", "M1", "Mx"],
                description="Metastasis stage (TNM)"
            ),
            Constraint(
                variable="surgical_approach",
                constraint_type="categorical",
                allowed_values=[
                    "Whipple", "distal_pancreatectomy", "total_pancreatectomy",
                    "palliative_bypass", "exploratory_laparotomy", "no_surgery"
                ],
                description="Type of surgical procedure"
            ),
            Constraint(
                variable="resection_margin",
                constraint_type="categorical",
                allowed_values=["R0", "R1", "R2", "unknown"],
                description="Surgical margin status"
            ),
            Constraint(
                variable="tumor_differentiation",
                constraint_type="categorical",
                allowed_values=["well", "moderate", "poor", "undifferentiated"],
                description="Tumor differentiation grade"
            ),
            Constraint(
                variable="CA19_9",
                constraint_type="range",
                value_range=(0, 100000),
                description="CA19-9 tumor marker (U/mL)"
            ),
            Constraint(
                variable="CEA",
                constraint_type="range",
                value_range=(0, 10000),
                description="CEA tumor marker (ng/mL)"
            ),
            Constraint(
                variable="ECOG",
                constraint_type="categorical",
                allowed_values=[0, 1, 2, 3, 4],
                description="ECOG performance status"
            ),
        ]

        for constraint in default_constraints:
            self.add_constraint(constraint)

    def get_imputation_history(self) -> List[ImputationResult]:
        """Get history of all imputations performed."""
        return self._imputation_history.copy()

    def get_confidence_summary(self) -> pd.DataFrame:
        """
        Get summary of confidence scores across all imputations.

        Returns
        -------
        pd.DataFrame
            Summary statistics by variable.
        """
        if not self._imputation_history:
            return pd.DataFrame()

        rows = []
        for result in self._imputation_history:
            rows.append({
                "variable": result.variable,
                "imputed_value": result.imputed_value,
                "confidence_score": result.confidence_score,
                "confidence_level": result.confidence_level.value,
                "constraints_satisfied": result.constraints_satisfied
            })

        return pd.DataFrame(rows)

    def filter_high_confidence(
        self,
        threshold: Optional[float] = None
    ) -> List[ImputationResult]:
        """
        Filter imputation history for high confidence results.

        Parameters
        ----------
        threshold : float, optional
            Confidence threshold (default: self.default_confidence_threshold).

        Returns
        -------
        list
            High confidence imputation results.
        """
        threshold = threshold or self.default_confidence_threshold
        return [
            r for r in self._imputation_history
            if r.confidence_score >= threshold
        ]

    # ------------------------------------------------------------------ #
    # Internal Logic
    # ------------------------------------------------------------------ #

    def _build_imputation_prompt(
        self,
        variable: str,
        patient_data: pd.Series,
        clinical_context: Optional[ClinicalContext],
        constraint: Optional[Constraint]
    ) -> str:
        """Build the prompt for LLM imputation."""
        prompt_parts = [
            "You are a clinical data imputation assistant specializing in pancreatic ductal adenocarcinoma (PDAC).",
            "",
            f"Task: Impute the missing value for variable '{variable}'.",
            ""
        ]

        # Add constraint information
        if constraint:
            prompt_parts.append("Constraint:")
            if constraint.constraint_type == "categorical" and constraint.allowed_values:
                prompt_parts.append(f"  Allowed values: {constraint.allowed_values}")
            elif constraint.constraint_type == "range" and constraint.value_range:
                prompt_parts.append(
                    f"  Value range: [{constraint.value_range[0]}, {constraint.value_range[1]}]"
                )
            prompt_parts.append("")

        # Add patient data
        prompt_parts.append("Patient Data:")
        for col, val in patient_data.items():
            if pd.notna(val) and col != variable:
                prompt_parts.append(f"  {col}: {val}")
        prompt_parts.append("")

        # Add clinical context
        if clinical_context:
            prompt_parts.append("Clinical Context:")
            prompt_parts.append(clinical_context.to_prompt_context())
            prompt_parts.append("")

        # Add output format instructions
        prompt_parts.extend([
            "Please provide your response in the following JSON format:",
            "{",
            '  "imputed_value": <the imputed value>,',
            '  "confidence_score": <float between 0 and 1>,',
            '  "reasoning": "<clinical reasoning for this imputation>",',
            '  "evidence": ["<supporting evidence 1>", "<supporting evidence 2>"]',
            "}"
        ])

        return "\n".join(prompt_parts)

    def _call_llm_placeholder(
        self,
        prompt: str,
        variable: str,
        constraint: Optional[Constraint]
    ) -> Dict[str, Any]:
        """
        Placeholder for LLM API call.

        TODO: Replace with actual LLM API integration.
        """
        # This is a placeholder that returns a reasonable default
        # In production, this would call an actual LLM API

        if constraint and constraint.constraint_type == "categorical":
            if constraint.allowed_values:
                # Return first allowed value with medium confidence
                return {
                    "imputed_value": constraint.allowed_values[0],
                    "confidence_score": 0.6,
                    "reasoning": f"Placeholder imputation. Selected first allowed value for {variable}.",
                    "evidence": ["Based on constraint specification"]
                }

        # Default numeric imputation
        return {
            "imputed_value": 0.0,
            "confidence_score": 0.5,
            "reasoning": f"Placeholder imputation for {variable}.",
            "evidence": ["No specific evidence available"]
        }

    def _parse_llm_response(
        self,
        response: Dict[str, Any],
        variable: str,
        constraint: Optional[Constraint],
        return_alternatives: bool
    ) -> ImputationResult:
        """Parse LLM response into ImputationResult."""
        confidence_score = response.get("confidence_score", 0.5)

        # Determine confidence level
        if confidence_score > 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score > 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score > 0.3:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.UNCERTAIN

        result = ImputationResult(
            variable=variable,
            imputed_value=response.get("imputed_value"),
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            evidence=response.get("evidence", []),
            reasoning=response.get("reasoning", ""),
            alternative_values=[]
        )

        # Generate alternatives if requested
        if return_alternatives and constraint:
            result.alternative_values = self._generate_alternatives(
                result.imputed_value, constraint
            )

        return result

    def _generate_alternatives(
        self,
        primary_value: Any,
        constraint: Constraint
    ) -> List[Tuple[Any, float]]:
        """Generate alternative values with probabilities."""
        alternatives = []

        if constraint.constraint_type == "categorical" and constraint.allowed_values:
            for val in constraint.allowed_values:
                if val != primary_value:
                    # Assign lower probability to alternatives
                    prob = 0.3 / (len(constraint.allowed_values) - 1)
                    alternatives.append((val, prob))

        return alternatives

    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score > 0.8:
            return ConfidenceLevel.HIGH
        elif score > 0.5:
            return ConfidenceLevel.MEDIUM
        elif score > 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN


# ==============================================================================
# Unit Tests
# ==============================================================================

def _test_llm_constraints():
    """Unit tests for LLMConstraintLayer."""

    # Test 1: Constraint validation
    print("Testing Constraint validation...")
    cat_constraint = Constraint(
        variable="T_stage",
        constraint_type="categorical",
        allowed_values=["T1", "T2", "T3", "T4"]
    )
    assert cat_constraint.validate("T2")[0], "T2 should be valid"
    assert not cat_constraint.validate("T5")[0], "T5 should be invalid"

    range_constraint = Constraint(
        variable="CA19_9",
        constraint_type="range",
        value_range=(0, 1000)
    )
    assert range_constraint.validate(500)[0], "500 should be in range"
    assert not range_constraint.validate(1500)[0], "1500 should be out of range"

    # Test 2: ClinicalContext
    print("Testing ClinicalContext...")
    context = ClinicalContext(
        patient_id="P001",
        diagnosis="PDAC",
        age=65,
        stage="III",
        comorbidities=["diabetes", "hypertension"],
        relevant_labs={"CA19_9": 125.0, "CEA": 5.2}
    )
    prompt_context = context.to_prompt_context()
    assert "PDAC" in prompt_context, "Diagnosis should be in context"
    assert "diabetes" in prompt_context, "Comorbidities should be in context"
    assert "CA19_9" in prompt_context, "Labs should be in context"

    # Test 3: LLMConstraintLayer
    print("Testing LLMConstraintLayer...")
    constraints = {
        "T_stage": cat_constraint,
        "CA19_9": range_constraint
    }
    layer = LLMConstraintLayer(constraints)

    # Test imputation (placeholder)
    patient_data = pd.Series({
        "age": 65,
        "gender": "M",
        "CA19_9": 125.0
    })
    result = layer.impute("T_stage", patient_data, context)
    assert isinstance(result, ImputationResult), "Should return ImputationResult"
    assert result.variable == "T_stage", "Variable name should match"
    assert 0 <= result.confidence_score <= 1, "Confidence should be in [0, 1]"

    # Test 4: Default PDAC constraints
    print("Testing default PDAC constraints...")
    layer2 = LLMConstraintLayer()
    layer2.register_default_pdac_constraints()
    assert "T_stage" in layer2.constraints, "T_stage constraint should be registered"
    assert "N_stage" in layer2.constraints, "N_stage constraint should be registered"
    assert "surgical_approach" in layer2.constraints, "surgical_approach constraint should be registered"

    # Test 5: Batch imputation
    print("Testing batch imputation...")
    results = layer2.impute_batch(["T_stage", "N_stage"], patient_data, context)
    assert len(results) == 2, "Should return 2 results"
    assert "T_stage" in results, "Should have T_stage result"
    assert "N_stage" in results, "Should have N_stage result"

    # Test 6: Confidence summary
    print("Testing confidence summary...")
    summary = layer2.get_confidence_summary()
    assert isinstance(summary, pd.DataFrame), "Should return DataFrame"
    assert len(summary) == 2, "Should have 2 imputation records"

    # Test 7: High confidence filter
    print("Testing high confidence filter...")
    high_conf = layer2.filter_high_confidence(threshold=0.5)
    assert isinstance(high_conf, list), "Should return list"

    # Test 8: JSON serialization
    print("Testing JSON serialization...")
    json_str = result.to_json()
    assert isinstance(json_str, str), "Should return string"
    parsed = json.loads(json_str)
    assert "imputed_value" in parsed, "JSON should have imputed_value"
    assert "confidence_score" in parsed, "JSON should have confidence_score"

    print("All LLM Constraints tests passed!")
    return True


if __name__ == "__main__":
    _test_llm_constraints()

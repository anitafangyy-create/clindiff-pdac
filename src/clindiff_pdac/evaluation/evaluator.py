"""
Evaluator Module - Simplified without sklearn dependency
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MSE"""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE"""
    return np.mean(np.abs(y_true - y_pred))


class ImputationEvaluator:
    """
    Evaluate imputation quality using various metrics
    """
    
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor
        self.metrics = {}
    
    def compute_metrics(
        self,
        imputed_values: np.ndarray,
        true_values: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute imputation metrics
        
        Args:
            imputed_values: Imputed values [n_samples, n_features]
            true_values: Ground truth values [n_samples, n_features]
            mask: Mask where 1 = was missing (evaluate here), 0 = observed
            
        Returns:
            Dictionary of metrics
        """
        # Only evaluate on originally missing values
        imputed_missing = imputed_values[mask == 1]
        true_missing = true_values[mask == 1]
        
        if len(imputed_missing) == 0:
            print("Warning: No missing values to evaluate")
            return {}
        
        metrics = {}
        
        # RMSE
        metrics['RMSE'] = root_mean_squared_error(true_missing, imputed_missing)
        
        # MAE
        metrics['MAE'] = mean_absolute_error(true_missing, imputed_missing)
        
        # MAPE (avoid division by zero)
        non_zero = true_missing != 0
        if non_zero.sum() > 0:
            mape = np.mean(np.abs((true_missing[non_zero] - imputed_missing[non_zero]) 
                                  / true_missing[non_zero])) * 100
            metrics['MAPE'] = mape
        
        # R²
        ss_res = np.sum((true_missing - imputed_missing) ** 2)
        ss_tot = np.sum((true_missing - true_missing.mean()) ** 2)
        metrics['R2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Correlation
        if len(imputed_missing) > 1:
            correlation = np.corrcoef(imputed_missing, true_missing)[0, 1]
            metrics['Correlation'] = correlation if not np.isnan(correlation) else 0
        
        self.metrics = metrics
        return metrics
    
    def compute_per_feature_metrics(
        self,
        imputed_values: np.ndarray,
        true_values: np.ndarray,
        mask: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Compute metrics per feature
        """
        results = []
        
        for i, feature in enumerate(feature_names):
            if i >= imputed_values.shape[1]:
                break
                
            imp = imputed_values[:, i]
            true = true_values[:, i]
            m = mask[:, i]
            
            if m.sum() == 0:
                continue
            
            metrics = self.compute_metrics(
                imp.reshape(-1, 1),
                true.reshape(-1, 1),
                m.reshape(-1, 1)
            )
            metrics['feature'] = feature
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def print_summary(self):
        """Print summary of metrics"""
        print("\n=== Imputation Evaluation Summary ===")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")


class ClinicalValidator:
    """
    Validate imputed values against clinical constraints
    """
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.violations = []
    
    def validate_physiological_ranges(
        self,
        values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Check if imputed values are within physiological ranges
        
        Returns:
            Dictionary with violation rates per feature
        """
        violation_rates = {}
        
        for i, feature in enumerate(feature_names):
            if i >= values.shape[1]:
                break
            
            if feature not in self.kg.constraints:
                continue
            
            constraint = self.kg.constraints[feature]
            feature_values = values[:, i]
            
            violations = 0
            total = len(feature_values)
            
            if 'min' in constraint:
                violations += (feature_values < constraint['min']).sum()
            if 'max' in constraint:
                violations += (feature_values > constraint['max']).sum()
            
            violation_rate = violations / total if total > 0 else 0
            violation_rates[feature] = violation_rate
        
        return violation_rates
    
    def validate_logical_consistency(
        self,
        values: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict]:
        """
        Check logical consistency between related variables
        
        Returns:
            List of detected inconsistencies
        """
        inconsistencies = []
        
        # Example: Total bilirubin should be >= direct bilirubin
        if 'bilirubin_total' in feature_names and 'direct_bilirubin' in feature_names:
            total_idx = feature_names.index('bilirubin_total')
            direct_idx = feature_names.index('direct_bilirubin')
            
            if total_idx < values.shape[1] and direct_idx < values.shape[1]:
                total_vals = values[:, total_idx]
                direct_vals = values[:, direct_idx]
                
                inconsistent = direct_vals > total_vals
                if inconsistent.any():
                    inconsistencies.append({
                        'rule': 'direct_bilirubin <= total_bilirubin',
                        'count': int(inconsistent.sum()),
                        'rate': float(inconsistent.mean())
                    })
        
        # Example: CA19-9 should be positive
        ca19_names = ['ca19_9', 'ca19-9', 'CA19-9']
        for ca19_name in ca19_names:
            if ca19_name in feature_names:
                ca199_idx = feature_names.index(ca19_name)
                if ca199_idx < values.shape[1]:
                    ca199_vals = values[:, ca199_idx]
                    
                    negative = ca199_vals < 0
                    if negative.any():
                        inconsistencies.append({
                            'rule': 'CA19-9 >= 0',
                            'count': int(negative.sum()),
                            'rate': float(negative.mean())
                        })
                break
        
        self.violations.extend(inconsistencies)
        return inconsistencies
    
    def validate_temporal_consistency(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Check temporal smoothness of imputed time series
        
        Returns:
            Dictionary with smoothness metrics
        """
        smoothness_scores = {}
        
        for i, feature in enumerate(feature_names):
            if i >= values.shape[1]:
                break
            
            feature_values = values[:, i]
            
            # Compute first differences
            diffs = np.diff(feature_values)
            
            # Coefficient of variation of differences
            if diffs.std() > 0:
                cv = diffs.std() / (np.abs(diffs).mean() + 1e-8)
                smoothness_scores[feature] = cv
        
        return smoothness_scores
    
    def generate_validation_report(
        self,
        imputed_values: np.ndarray,
        feature_names: List[str],
        timestamps: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate comprehensive validation report
        """
        report = {
            'physiological_violations': self.validate_physiological_ranges(
                imputed_values, feature_names
            ),
            'logical_inconsistencies': self.validate_logical_consistency(
                imputed_values, feature_names
            )
        }
        
        if timestamps is not None:
            report['temporal_consistency'] = self.validate_temporal_consistency(
                imputed_values, timestamps, feature_names
            )
        
        # Overall statistics
        total_violations = sum(
            1 for v in report['physiological_violations'].values() if v > 0
        )
        report['summary'] = {
            'features_with_violations': total_violations,
            'total_features': len(feature_names),
            'violation_rate': total_violations / len(feature_names) if feature_names else 0
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print validation report"""
        print("\n=== Clinical Validation Report ===")
        
        print("\nPhysiological Range Violations:")
        for feature, rate in report['physiological_violations'].items():
            if rate > 0:
                print(f"  {feature}: {rate*100:.2f}%")
        
        print("\nLogical Inconsistencies:")
        for inconsistency in report['logical_inconsistencies']:
            print(f"  {inconsistency['rule']}: {inconsistency['count']} cases "
                  f"({inconsistency['rate']*100:.2f}%)")
        
        if 'summary' in report:
            print(f"\nSummary: {report['summary']['features_with_violations']}/"
                  f"{report['summary']['total_features']} features have violations")


def compare_methods(
    methods_results: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple imputation methods
    
    Args:
        methods_results: Dict of {method_name: metrics_dict}
        metrics_to_plot: List of metrics to include
        
    Returns:
        Comparison DataFrame
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2']
    
    comparison = []
    for method, metrics in methods_results.items():
        row = {'Method': method}
        for metric in metrics_to_plot:
            row[metric] = metrics.get(metric, np.nan)
        comparison.append(row)
    
    return pd.DataFrame(comparison)

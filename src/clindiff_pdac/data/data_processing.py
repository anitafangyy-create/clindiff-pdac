"""
Data Processing Module for EMR Imputation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class DataPreprocessor:
    """
    Preprocessor for EMR data
    """
    
    def __init__(
        self,
        feature_names: List[str],
        categorical_features: Optional[List[str]] = None,
        normalization_method: str = "standard"
    ):
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.normalization_method = normalization_method
        
        self.mean = {}
        self.std = {}
        self.min = {}
        self.max = {}
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            df: DataFrame with features
        """
        for col in self.feature_names:
            if col in df.columns:
                data = df[col].dropna()
                
                if col in self.categorical_features:
                    self.min[col] = 0
                    self.max[col] = 1
                else:
                    if self.normalization_method == "standard":
                        self.mean[col] = data.mean()
                        self.std[col] = data.std() + 1e-8
                    elif self.normalization_method == "minmax":
                        self.min[col] = data.min()
                        self.max[col] = data.max()
                        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform data to tensors
        
        Args:
            df: DataFrame with features
            
        Returns:
            (values, mask) tensors
        """
        values = []
        masks = []
        
        for col in self.feature_names:
            if col in df.columns:
                col_data = df[col].values
                
                # Create mask (1 = observed, 0 = missing)
                mask = (~pd.isna(col_data)).astype(np.float32)
                
                # Normalize
                if col in self.categorical_features:
                    norm_data = col_data
                elif self.normalization_method == "standard":
                    norm_data = (col_data - self.mean.get(col, 0)) / self.std.get(col, 1)
                else:
                    range_val = self.max.get(col, 1) - self.min.get(col, 0)
                    if range_val > 0:
                        norm_data = (col_data - self.min.get(col, 0)) / range_val
                    else:
                        norm_data = col_data
                
                # Replace NaN with 0
                norm_data = np.nan_to_num(norm_data, nan=0.0)
                values.append(norm_data)
                masks.append(mask)
            else:
                values.append(np.zeros(len(df)))
                masks.append(np.zeros(len(df)))
        
        values_tensor = torch.tensor(np.stack(values, axis=1), dtype=torch.float32)
        mask_tensor = torch.tensor(np.stack(masks, axis=1), dtype=torch.float32)
        
        return values_tensor, mask_tensor
    
    def inverse_transform(self, values: torch.Tensor, mask: torch.Tensor) -> pd.DataFrame:
        """
        Inverse transform normalized values back to original scale
        
        Args:
            values: Normalized values tensor
            mask: Missing mask tensor
            
        Returns:
            DataFrame with original scale values
        """
        result = {}
        
        for i, col in enumerate(self.feature_names):
            col_values = values[:, i].numpy()
            
            if col in self.categorical_features:
                result[col] = col_values
            elif self.normalization_method == "standard":
                result[col] = col_values * self.std.get(col, 1) + self.mean.get(col, 0)
            else:
                range_val = self.max.get(col, 1) - self.min.get(col, 0)
                result[col] = col_values * range_val + self.min.get(col, 0)
        
        return pd.DataFrame(result)
    
    def save(self, path: str):
        """Save preprocessor state"""
        state = {
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'normalization_method': self.normalization_method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load preprocessor state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        preprocessor = cls(
            feature_names=state['feature_names'],
            categorical_features=state['categorical_features'],
            normalization_method=state['normalization_method']
        )
        preprocessor.mean = state.get('mean', {})
        preprocessor.std = state.get('std', {})
        preprocessor.min = state.get('min', {})
        preprocessor.max = state.get('max', {})
        
        return preprocessor


class PDACDataset(Dataset):
    """
    Dataset for PDAC EMR data with missing values
    """
    
    def __init__(
        self,
        data_path: str,
        preprocessor: Optional[DataPreprocessor] = None,
        max_seq_len: int = 50,
        patient_id_col: str = 'patient_id',
        time_col: Optional[str] = None
    ):
        self.df = pd.read_csv(data_path)
        self.preprocessor = preprocessor
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        
        if preprocessor is not None:
            self.values, self.mask = preprocessor.transform(self.df)
        else:
            raise ValueError("Preprocessor is required")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patient's data
        """
        return {
            'values': self.values[idx],
            'mask': self.mask[idx],
            'patient_id': self.df.iloc[idx][self.patient_id_col]
        }


class EMRDataLoader:
    """
    Custom data loader for EMR data with batch generation
    """
    
    def __init__(
        self,
        dataset: PDACDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)
            
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        batch_indices = self.indices[
            self.current_idx:self.current_idx + self.batch_size
        ]
        self.current_idx += self.batch_size
        
        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration
        
        # Gather batch data
        batch_values = self.dataset.values[batch_indices]
        batch_mask = self.dataset.mask[batch_indices]
        batch_ids = [self.dataset.df.iloc[i]['patient_id'] 
                     for i in batch_indices]
        
        return {
            'values': batch_values,
            'mask': batch_mask,
            'patient_ids': batch_ids
        }
    
    def __len__(self) -> int:
        num_batches = len(self.indices) // self.batch_size
        if not self.drop_last and len(self.indices) % self.batch_size > 0:
            num_batches += 1
        return num_batches


def create_missing_data(
    data: np.ndarray,
    missing_rate: float,
    mechanism: str = "MCAR"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create missing data for evaluation
    
    Args:
        data: Complete data array
        missing_rate: Rate of missingness (0-1)
        mechanism: MCAR, MAR, or MNAR
        
    Returns:
        (incomplete_data, true_missing_mask)
    """
    if mechanism == "MCAR":
        mask = np.random.rand(*data.shape) > missing_rate
        
    elif mechanism == "MAR":
        # Missing depends on other observed values
        prob = missing_rate * (1 + data[:, :5].mean(axis=1) / 10)  # Simplified
        mask = np.random.rand(*data.shape) > prob[:, None]
        
    elif mechanism == "MNAR":
        # Missing depends on the value itself
        prob = missing_rate * (1 + np.abs(data) / data.std())
        mask = np.random.rand(*data.shape) > np.clip(prob, 0, 1)
    
    incomplete_data = np.where(mask, data, np.nan)
    
    # Return: mask is True where data is observed (for consistency)
    # Also return a separate mask for missing positions
    return incomplete_data, mask


def load_pancreatic_cancer_data(
    data_path: str,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load and preprocess pancreatic cancer EMR data
    
    Args:
        data_path: Path to CSV file
        feature_cols: Optional list of feature columns to use
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(data_path)
    
    if feature_cols is not None:
        available_cols = [c for c in feature_cols if c in df.columns]
        df = df[available_cols]
    
    # Basic preprocessing
    # Convert string columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df


def get_default_pancreatic_features() -> List[str]:
    """
    Get default feature columns for pancreatic cancer EMR
    """
    return [
        'age',
        'CA19-9', 'CEA', 'CA125',
        'bilirubin_total', 'ALT', 'AST', 'ALP', 'GGT',
        'amylase', 'lipase',
        'glucose', 'HbA1c',
        'albumin', 'hemoglobin',
        'has_diabetes', 'has_jaundice', 'has_weight_loss',
        'has_abdominal_pain', 'has_nausea'
    ]

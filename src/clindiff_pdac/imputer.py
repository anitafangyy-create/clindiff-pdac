"""
ClinDiff-PDAC: Clinical Differentiation for Pancreatic Ductal Adenocarcinoma

Main API for feature correlation imputation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Union


class ClinDiffPDAC:
    """
    ClinDiff-PDAC imputer with feature correlation modeling.
    
    Sklearn-style API for easy integration.
    """
    
    def __init__(
        self,
        n_features: int = None,
        embedding_dim: int = 64,
        diffusion_steps: int = 200,
        feature_correlation_weight: float = 0.3,
        num_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str = None,
        random_state: int = 42
    ):
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.diffusion_steps = diffusion_steps
        self.feature_correlation_weight = feature_correlation_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        
        self.model_ = None
        self.feature_means_ = None
        self.feature_stds_ = None
        self.fitted_ = False
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self):
        """Build the neural network model."""
        from .models.diffusion import KnowledgeGuidedDiffusion
        
        self.model_ = KnowledgeGuidedDiffusion(
            data_dim=self.n_features,
            kg_dim=self.embedding_dim,
            hidden_dims=[128, 128, 64],
            num_timesteps=self.diffusion_steps
        ).to(self.device)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """
        Fit the imputer on data with missing values.
        
        Args:
            X: Data with missing values (numpy array or DataFrame)
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.astype(np.float32)
        self.n_features = X.shape[1]
        
        # Normalize
        self.feature_means_ = np.nanmean(X, axis=0)
        self.feature_stds_ = np.nanstd(X, axis=0)
        self.feature_stds_[self.feature_stds_ == 0] = 1.0
        
        X_normalized = (X - self.feature_means_) / self.feature_stds_
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        
        # Build model
        self._build_model()
        
        # Create mask
        mask = ~np.isnan(X)
        
        # Training
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        n_samples = len(X_normalized)
        
        for epoch in range(self.num_epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_data = torch.tensor(X_normalized[batch_idx], dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(mask[batch_idx], dtype=torch.float32).to(self.device)
                
                # Generate random knowledge context
                kg_context = torch.randn(len(batch_idx), self.embedding_dim).to(self.device)
                
                # Compute loss
                loss_dict = self.model_.compute_loss(batch_data, kg_context, batch_mask)
                loss = loss_dict['total_loss']
                
                # Skip if loss is NaN
                if torch.isnan(loss):
                    print(f"     Warning: NaN loss at epoch {epoch + 1}, skipping batch")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 25 == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else float('nan')
                print(f"     Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        self.fitted_ = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Impute missing values in X.
        
        Args:
            X: Data with missing values
            
        Returns:
            Imputed data (same type as input)
        """
        if not self.fitted_:
            raise RuntimeError("Imputer not fitted. Call fit() first.")
        
        # Convert to numpy
        is_dataframe = isinstance(X, pd.DataFrame)
        columns = X.columns if is_dataframe else None
        index = X.index if is_dataframe else None
        
        X = X.values if is_dataframe else X
        X = X.astype(np.float32)
        
        # Normalize
        X_normalized = (X - self.feature_means_) / self.feature_stds_
        mask = ~np.isnan(X)
        X_filled = np.nan_to_num(X_normalized, nan=0.0)
        
        # Impute using diffusion
        self.model_.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(X_filled, dtype=torch.float32).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(self.device)
            kg_context = torch.randn(len(X), self.embedding_dim).to(self.device)
            
            # Simple forward diffusion and denoising
            imputed_normalized = self._impute_batch(data_tensor, mask_tensor, kg_context)
        
        # Denormalize
        imputed = imputed_normalized * self.feature_stds_ + self.feature_means_
        
        # Restore original observed values
        imputed[mask] = X[mask]
        
        # Convert back to DataFrame if needed
        if is_dataframe:
            imputed = pd.DataFrame(imputed, columns=columns, index=index)
        
        return imputed
    
    def _impute_batch(self, data, mask, kg_context):
        """Fast imputation using the diffusion model."""
        # Use iterative denoising with fewer steps for speed
        # Start with mean imputation
        imputed = data.clone().detach()
        
        # Fewer diffusion steps for speed
        n_steps = min(20, self.diffusion_steps)
        step_size = self.diffusion_steps // n_steps
        
        # Iterative refinement using the denoiser
        for step in range(n_steps):
            t = self.diffusion_steps - step * step_size
            
            # Add noise then denoise
            if step > 0:
                noise_level = 0.1 * (step / n_steps)
                noise = torch.randn_like(imputed) * noise_level
                noisy = imputed + noise
            else:
                noisy = imputed
            
            # Predict noise
            t_batch = torch.full((imputed.shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model_.denoiser(noisy, t_batch, kg_context)
            
            # Update estimate
            alpha = 0.3
            imputed = imputed * (1 - alpha) + (noisy - alpha * predicted_noise)
            
            # Keep observed values fixed
            imputed = imputed * (1 - mask) + data * mask
        
        return imputed.cpu().numpy()
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

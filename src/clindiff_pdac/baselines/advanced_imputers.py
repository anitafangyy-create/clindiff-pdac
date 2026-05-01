"""
Advanced AI imputation baselines for EHR data.
Implements: GAIN, MIDA, simple attention-based methods.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class GAINGenerator(nn.Module):
    """GAIN: Generative Adversarial Imputation Network"""
    
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        self.n_features = n_features
        
        # Generator
        self.gen = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_features)
        )
        
    def forward(self, x, mask):
        """Forward pass"""
        x_mask = torch.cat([x, mask], dim=1)
        generated_data = self.gen(x_mask)
        x_imputed = x * mask + generated_data * (1 - mask)
        return x_imputed
    
    def loss(self, x, mask):
        x_imputed = self.forward(x, mask)
        loss = torch.mean(mask * (x_imputed - x) ** 2)
        return loss
    
    def impute(self, x, mask):
        self.eval()
        with torch.no_grad():
            return self.forward(x, mask)


class MIDA(nn.Module):
    """MIDA: Missing Data Imputation with Denoising Autoencoders"""
    
    def __init__(self, n_features, hidden_dims=[128, 64]):
        super().__init__()
        bottleneck = hidden_dims[-1]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], bottleneck),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], n_features)
        )
        
    def forward(self, x, mask):
        # Concatenate data and mask
        x_mask = torch.cat([x, mask], dim=1)
        
        # Encode
        z = self.encoder(x_mask)
        
        # Decode
        imputed = self.decoder(z)
        
        # Combine with observed values
        x_imputed = x * mask + imputed * (1 - mask)
        
        return x_imputed
    
    def loss(self, x, mask):
        x_imputed = self.forward(x, mask)
        loss = torch.mean(mask * (x_imputed - x) ** 2)
        return loss
    
    def impute(self, x, mask):
        self.eval()
        with torch.no_grad():
            return self.forward(x, mask)


class AttentionImputer(nn.Module):
    """Simple attention-based imputation"""
    
    def __init__(self, n_features, d_model=64, n_heads=4):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Input embedding
        self.embed = nn.Linear(n_features, d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=0.1)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_features)
        )
        
    def forward(self, x, mask):
        # Embed
        x_embedded = self.embed(x).unsqueeze(0)  # [1, batch, d_model]
        
        # Self-attention
        attn_output, _ = self.attention(x_embedded, x_embedded, x_embedded)
        
        # Output
        imputed = self.output(attn_output.squeeze(0))
        
        # Combine with observed values
        x_imputed = x * mask + imputed * (1 - mask)
        
        return x_imputed
    
    def loss(self, x, mask):
        x_imputed = self.forward(x, mask)
        loss = torch.mean(mask * (x_imputed - x) ** 2)
        return loss
    
    def impute(self, x, mask):
        self.eval()
        with torch.no_grad():
            return self.forward(x, mask)


def train_model(model, train_data, train_mask, device, epochs=200, lr=1e-3):
    """Train any model with standard procedure and gradient clipping"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_tensor = torch.tensor(train_data, device=device)
    mask_tensor = torch.tensor(train_mask, device=device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(train_tensor, mask_tensor)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"     Warning: NaN loss at epoch {epoch+1}, skipping")
            continue
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    return model


def get_all_baselines():
    """Return all available baseline models"""
    return {
        'GAIN': GAINGenerator,
        'MIDA': MIDA,
        'Attention': AttentionImputer,
    }

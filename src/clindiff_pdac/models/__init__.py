"""
Core ClinDiff-PDAC Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class Time2Vec(nn.Module):
    """Time2Vec encoding for irregular time series"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, embed_dim - 1))
        self.b = nn.Parameter(torch.randn(1, embed_dim - 1))
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch, seq_len] - timestamps
        Returns:
            [batch, seq_len, embed_dim]
        """
        t = t.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Linear component
        v0 = self.w0 * t + self.b0
        
        # Periodic component
        v = torch.sin(self.w * t + self.b)
        
        return torch.cat([v0, v], dim=-1)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for local pattern extraction"""
    
    def __init__(self, input_dim: int, channel_sizes: List[int], kernel_size: int = 3):
        super().__init__()
        layers = []
        num_levels = len(channel_sizes)
        
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else channel_sizes[i - 1]
            out_channels = channel_sizes[i]
            
            # Causal convolution with dilation
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, channels, seq_len]
        """
        seq_len = x.size(-1)
        out = self.network(x)
        return out[..., :seq_len]


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for structured EMR data (labs, vitals)
    Combines Transformer for global dependencies and TCN for local patterns
    """
    
    def __init__(
        self, 
        input_dim: int, 
        d_model: int = 256, 
        nhead: int = 8, 
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Time-aware embedding
        self.time_embed = Time2Vec(d_model)
        
        # Value embedding with missing indicator
        self.value_embed = nn.Linear(input_dim * 2, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        self._transformer_batch_first = True
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
        except TypeError:
            self._transformer_batch_first = False
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # TCN for local patterns
        self.tcn = TemporalConvNet(d_model, [d_model // 2, d_model // 2, d_model])
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features] - normalized values
            mask: [batch, seq_len, features] - 1 if observed, 0 if missing
            timestamps: [batch, seq_len] - actual time points
            
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Time encoding [batch, seq_len, d_model]
        time_encoding = self.time_embed(timestamps)
        
        # Value encoding with missing indicator
        x_augmented = torch.cat([x * mask, mask], dim=-1)
        value_encoding = self.value_embed(x_augmented)
        
        # Combine time and value
        h = value_encoding + time_encoding
        h = self.pos_encoder(h)
        
        # Transformer for global dependencies
        if self._transformer_batch_first:
            h_global = self.transformer(h)
        else:
            h_global = self.transformer(h.transpose(0, 1)).transpose(0, 1)
        
        # TCN for local patterns
        h_tcn_input = h.transpose(1, 2)  # [batch, d_model, seq_len]
        h_local = self.tcn(h_tcn_input).transpose(1, 2)  # [batch, seq_len, d_model]
        
        # Concatenate global and local
        h_combined = torch.cat([h_global, h_local], dim=-1)
        
        return self.output_proj(h_combined)


class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder combining structured data and clinical text
    """
    
    def __init__(
        self,
        temporal_dim: int,
        text_dim: int = 768,
        output_dim: int = 256,
        num_text_layers: int = 4
    ):
        super().__init__()
        
        # Temporal encoder for structured data
        self.temporal_encoder = TemporalEncoder(temporal_dim, d_model=output_dim)
        
        # Text encoder (simplified - would use Bio_ClinicalBERT in practice)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=8,
                dim_feedforward=text_dim * 4,
                batch_first=True
            ),
            num_layers=num_text_layers
        )
        
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        structured_data: Dict[str, torch.Tensor],
        text_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            structured_data: dict with 'values', 'mask', 'timestamps'
            text_data: [batch, seq_len, text_dim] - clinical notes embedding
            
        Returns:
            [batch, seq_len, output_dim]
        """
        # Encode structured data
        temporal_encoded = self.temporal_encoder(
            structured_data['values'],
            structured_data['mask'],
            structured_data['timestamps']
        )
        
        if text_data is not None:
            # Encode text
            text_encoded = self.text_encoder(text_data)
            text_encoded = self.text_proj(text_encoded)
            
            # Cross-modal attention
            attended, _ = self.cross_attention(
                query=temporal_encoded,
                key=text_encoded,
                value=text_encoded
            )
            
            # Fusion
            combined = torch.cat([temporal_encoded, attended], dim=-1)
            return self.fusion(combined)
        
        return temporal_encoded


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

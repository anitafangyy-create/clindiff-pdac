"""
Knowledge-Guided Diffusion Model for EMR Imputation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math


class KnowledgeGuidedDiffusion(nn.Module):
    """
    Conditional diffusion model with clinical knowledge constraints
    """
    
    def __init__(
        self,
        data_dim: int,
        kg_dim: int = 256,
        hidden_dims: List[int] = [512, 512, 256],
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine"
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.kg_dim = kg_dim
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        self.betas = self._get_beta_schedule(beta_schedule, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Denoising network (U-Net style)
        self.denoiser = ConditionalDenoiser(
            data_dim=data_dim,
            kg_dim=kg_dim,
            hidden_dims=hidden_dims,
            time_embed_dim=128
        )
        
        # Knowledge-guided constraint network
        self.constraint_net = KnowledgeConstraintNetwork(
            data_dim=data_dim,
            kg_dim=kg_dim
        )
        
    def _get_beta_schedule(self, schedule: str, timesteps: int) -> torch.Tensor:
        """Get noise schedule"""
        if schedule == "linear":
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            return self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as in Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_diffusion(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x0: [batch, data_dim] - original data
            t: [batch] - timesteps
            noise: optional noise to use
            
        Returns:
            xt: noised data
            noise: the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Get schedule values for timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # Forward diffusion
        xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
        
        return xt, noise
    
    def reverse_diffusion(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        kg_context: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        x_observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        
        Args:
            xt: [batch, data_dim] - current noisy data
            t: [batch] - current timestep
            kg_context: [batch, kg_dim] - knowledge graph context
            observed_mask: [batch, data_dim] - 1 for observed values
            x_observed: [batch, data_dim] - original observed values
            
        Returns:
            x_{t-1}: denoised data
        """
        # Predict noise
        predicted_noise = self.denoiser(xt, t, kg_context)
        
        # Compute x_0 prediction
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        x0_pred = (xt - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        # Apply clinical constraints
        x0_pred = self.apply_clinical_constraints(x0_pred, kg_context)
        
        # Keep observed values
        if observed_mask is not None and x_observed is not None:
            x0_pred = observed_mask * x_observed + (1 - observed_mask) * x0_pred
        
        # Compute x_{t-1}
        if t[0] > 0:
            alpha_t_prev = self.alphas_cumprod[t - 1].view(-1, 1)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            
            # Reparameterization
            mean = sqrt_alpha_t_prev * x0_pred
            variance = 1 - alpha_t_prev
            
            noise = torch.randn_like(xt) if t[0] > 0 else torch.zeros_like(xt)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = x0_pred
        
        return x_prev
    
    def apply_clinical_constraints(
        self, 
        x: torch.Tensor, 
        kg_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply soft clinical constraints based on knowledge graph
        """
        # Get constraint corrections from knowledge graph
        constraint_weights, constraint_directions = self.constraint_net(x, kg_context)
        
        # Apply soft constraints
        x_corrected = x + constraint_weights * constraint_directions
        
        return x_corrected
    
    def sample(
        self,
        shape: Tuple[int, ...],
        kg_context: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        x_observed: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate samples using the reverse diffusion process
        
        Args:
            shape: shape of the data to generate
            kg_context: knowledge graph context
            observed_mask: mask for observed values
            x_observed: observed values
            device: device to use
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.reverse_diffusion(
                x, t_batch, kg_context, observed_mask, x_observed
            )
        
        return x
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        kg_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss
        
        Args:
            x0: [batch, data_dim] - original data
            kg_context: [batch, kg_dim] - knowledge graph context
            mask: [batch, data_dim] - observed value mask
            
        Returns:
            Dictionary of losses
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion
        noise = torch.randn_like(x0)
        xt, _ = self.forward_diffusion(x0, t, noise)
        
        # Predict noise
        predicted_noise = self.denoiser(xt, t, kg_context)
        
        # MSE loss on observed values only
        if mask is not None:
            loss = F.mse_loss(predicted_noise * mask, noise * mask)
        else:
            loss = F.mse_loss(predicted_noise, noise)
        
        # Knowledge constraint loss
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)
        constraint_loss = self.constraint_net.compute_constraint_loss(x0_pred, kg_context)
        
        total_loss = loss + 0.1 * constraint_loss
        
        return {
            'total_loss': total_loss,
            'noise_loss': loss,
            'constraint_loss': constraint_loss
        }
    
    def predict_x0_from_noise(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x0 = (xt - sqrt_one_minus_alpha_t * noise) / sqrt_alpha_t
        return x0


class ConditionalDenoiser(nn.Module):
    """
    U-Net style denoising network conditioned on time and knowledge
    """
    
    def __init__(
        self,
        data_dim: int,
        kg_dim: int,
        hidden_dims: List[int],
        time_embed_dim: int = 128
    ):
        super().__init__()
        
        self.time_embed = SinusoidalPositionEmbeddings(time_embed_dim)
        
        # Time projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Knowledge projection
        self.kg_proj = nn.Linear(kg_dim, time_embed_dim)
        
        # Encoder
        layers = []
        prev_dim = data_dim + time_embed_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*layers)
        
        # Decoder
        # This implementation uses a plain MLP decoder without explicit skip
        # connections, so the layer widths should follow the encoded width only.
        decoder_dims = list(reversed(hidden_dims[:-1]))
        layers = []
        for dim in decoder_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, data_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        kg_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, data_dim] - noisy data
            t: [batch] - timesteps
            kg_context: [batch, kg_dim] - knowledge context
            
        Returns:
            [batch, data_dim] - predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        # Knowledge embedding
        kg_emb = self.kg_proj(kg_context)
        
        # Combine conditioning
        cond = t_emb + kg_emb
        
        # Concatenate input and conditioning
        h = torch.cat([x, cond], dim=-1)
        
        # Encode
        h = self.encoder(h)
        
        # Decode (simplified - full U-Net would have skip connections)
        noise_pred = self.decoder(h)
        
        return noise_pred


class KnowledgeConstraintNetwork(nn.Module):
    """
    Network to compute clinical knowledge-based constraints
    """
    
    def __init__(self, data_dim: int, kg_dim: int):
        super().__init__()
        
        self.constraint_mlp = nn.Sequential(
            nn.Linear(data_dim + kg_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Constraint weight (how much to apply)
        self.weight_head = nn.Sequential(
            nn.Linear(128, data_dim),
            nn.Sigmoid()
        )
        
        # Constraint direction (which direction to push)
        self.direction_head = nn.Linear(128, data_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        kg_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute constraint weights and directions
        
        Returns:
            weights: [batch, data_dim] - how much to apply constraint
            directions: [batch, data_dim] - direction of correction
        """
        h = torch.cat([x, kg_context], dim=-1)
        h = self.constraint_mlp(h)
        
        weights = self.weight_head(h)
        directions = self.direction_head(h)
        
        return weights, directions
    
    def compute_constraint_loss(
        self, 
        x: torch.Tensor, 
        kg_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss to encourage clinically plausible values
        """
        # This would check against clinical rules from KG
        # Simplified version - just encourage values in reasonable range
        
        # Penalize extreme values
        range_penalty = torch.mean(F.relu(torch.abs(x) - 3))
        
        return range_penalty


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings"""
    
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

"""
Unit tests for ClinDiff-PDAC
"""

import pytest
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clindiff_pdac.models.diffusion import (
    KnowledgeGuidedDiffusion,
    SinusoidalPositionEmbeddings
)
from clindiff_pdac.models.encoder import TemporalEncoder
from clindiff_pdac.knowledge_graph.knowledge_graph import (
    PDACKnowledgeGraph,
    KnowledgeGraphEncoder
)
from clindiff_pdac.data.data_processing import (
    DataPreprocessor,
    create_missing_data
)
from clindiff_pdac.evaluation.evaluator import (
    ImputationEvaluator,
    ClinicalValidator
)


class TestDiffusionModel:
    """Test diffusion model"""
    
    def test_diffusion_init(self):
        """Test model initialization"""
        model = KnowledgeGuidedDiffusion(
            data_dim=20,
            kg_dim=256,
            hidden_dims=[512, 512, 256],
            num_timesteps=100
        )
        
        assert model.data_dim == 20
        assert model.kg_dim == 256
        assert model.num_timesteps == 100
        
    def test_forward_diffusion(self):
        """Test forward diffusion process"""
        model = KnowledgeGuidedDiffusion(data_dim=10, num_timesteps=100)
        
        x0 = torch.randn(4, 10)
        t = torch.randint(0, 100, (4,))
        
        xt, noise = model.forward_diffusion(x0, t)
        
        assert xt.shape == x0.shape
        assert noise.shape == x0.shape
        
    def test_reverse_diffusion(self):
        """Test reverse diffusion process"""
        model = KnowledgeGuidedDiffusion(data_dim=10, num_timesteps=100)
        
        xt = torch.randn(4, 10)
        t = torch.tensor([50, 50, 50, 50])
        kg_context = torch.randn(4, 256)
        
        x_prev = model.reverse_diffusion(xt, t, kg_context)
        
        assert x_prev.shape == xt.shape
        
    def test_sample(self):
        """Test sampling"""
        model = KnowledgeGuidedDiffusion(data_dim=10, num_timesteps=50)
        
        samples = model.sample(
            shape=(2, 10),
            kg_context=torch.randn(2, 256),
            device='cpu'
        )
        
        assert samples.shape == (2, 10)
        
    def test_compute_loss(self):
        """Test loss computation"""
        model = KnowledgeGuidedDiffusion(data_dim=10, num_timesteps=100)
        
        x0 = torch.randn(4, 10)
        kg_context = torch.randn(4, 256)
        mask = torch.randint(0, 2, (4, 10)).float()
        
        losses = model.compute_loss(x0, kg_context, mask)
        
        assert 'total_loss' in losses
        assert 'noise_loss' in losses
        assert losses['total_loss'].item() >= 0


class TestTemporalEncoder:
    """Test temporal encoder"""
    
    def test_encoder_init(self):
        """Test encoder initialization"""
        encoder = TemporalEncoder(
            input_dim=20,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        
        assert encoder.d_model == 128
        
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        encoder = TemporalEncoder(
            input_dim=20,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 20)
        mask = torch.randint(0, 2, (batch_size, seq_len, 20)).float()
        timestamps = torch.randn(batch_size, seq_len)
        
        output = encoder(x, mask, timestamps)
        
        assert output.shape == (batch_size, seq_len, 128)


class TestKnowledgeGraph:
    """Test knowledge graph"""
    
    def test_kg_init(self):
        """Test knowledge graph initialization"""
        kg = PDACKnowledgeGraph()
        
        assert len(kg.entities) > 0
        assert len(kg.relations) > 0
        assert len(kg.constraints) > 0
        
    def test_get_neighbors(self):
        """Test getting neighboring entities"""
        kg = PDACKnowledgeGraph()
        
        neighbors = kg.get_neighbors('CA19-9')
        
        assert len(neighbors) > 0
        
    def test_check_constraint(self):
        """Test constraint checking"""
        kg = PDACKnowledgeGraph()
        
        # Valid value
        assert kg.check_constraint('CA19-9', 100) == True
        
        # Invalid value
        assert kg.check_constraint('CA19-9', -10) == False
        
        # Unknown variable
        assert kg.check_constraint('unknown', 100) == True
        
    def test_get_implication_rules(self):
        """Test getting implication rules"""
        kg = PDACKnowledgeGraph()
        
        rules = kg.get_implication_rules('biliary_obstruction')
        
        assert isinstance(rules, list)


class TestDataProcessing:
    """Test data processing"""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor(
            feature_names=['age', 'CA19-9'],
            normalization_method='standard'
        )
        
        assert len(preprocessor.feature_names) == 2
        
    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform"""
        df = pd.DataFrame({
            'age': np.random.normal(65, 10, 100),
            'CA19-9': np.random.lognormal(4, 1, 100)
        })
        
        preprocessor = DataPreprocessor(
            feature_names=['age', 'CA19-9'],
            normalization_method='standard'
        )
        preprocessor.fit(df)
        
        values, mask = preprocessor.transform(df)
        
        assert values.shape == (100, 2)
        assert mask.shape == (100, 2)
        
    def test_create_missing_data_mcar(self):
        """Test MCAR data creation"""
        data = np.random.randn(100, 10)
        missing_data, mask = create_missing_data(data, 0.3, 'MCAR')
        
        assert missing_data.shape == data.shape
        assert mask.shape == data.shape
        assert np.isnan(missing_data).mean() > 0.2


class TestEvaluation:
    """Test evaluation"""
    
    def test_imputation_evaluator(self):
        """Test imputation evaluator"""
        evaluator = ImputationEvaluator()
        
        true_values = np.random.randn(100, 10)
        imputed_values = true_values + np.random.randn(100, 10) * 0.1
        mask = np.random.randint(0, 2, (100, 10))
        
        metrics = evaluator.compute_metrics(imputed_values, true_values, mask)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        
    def test_clinical_validator(self):
        """Test clinical validator"""
        kg = PDACKnowledgeGraph()
        validator = ClinicalValidator(kg)
        
        values = np.random.randn(100, 5)
        feature_names = ['CA19-9', 'bilirubin_total', 'ALT', 'AST', 'ALP']
        
        violations = validator.validate_physiological_ranges(values, feature_names)
        
        assert isinstance(violations, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Example training script for ClinDiff-PDAC
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clindiff_pdac import (
    KnowledgeGuidedDiffusion,
    MultiModalEncoder,
    ClinDiffTrainer,
    TrainingConfig,
    DataPreprocessor,
    PDACDataset
)


def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """
    Create synthetic pancreatic cancer data for demonstration
    """
    np.random.seed(42)
    
    # Simulate correlated features (like real PDAC EMR data)
    # Age
    age = np.random.normal(65, 12, n_samples)
    
    # CA19-9 (strongly correlated with cancer stage)
    ca199 = np.random.lognormal(4, 1.5, n_samples)
    ca199 = np.clip(ca199, 0, 10000)
    
    # CEA
    cea = np.random.lognormal(2, 1, n_samples)
    cea = np.clip(cea, 0, 1000)
    
    # Bilirubin
    bilirubin = np.random.exponential(1.5, n_samples)
    bilirubin = np.clip(bilirubin, 0.1, 30)
    
    # Liver enzymes
    alt = np.random.exponential(50, n_samples) + 20
    ast = np.random.exponential(50, n_samples) + 20
    alp = np.random.exponential(150, n_samples) + 50
    
    # Glucose
    glucose = np.random.normal(120, 40, n_samples)
    glucose = np.clip(glucose, 50, 500)
    
    # Albumin
    albumin = np.random.normal(3.8, 0.6, n_samples)
    albumin = np.clip(albumin, 1.5, 5)
    
    # Hemoglobin
    hemoglobin = np.random.normal(12, 2, n_samples)
    hemoglobin = np.clip(hemoglobin, 5, 18)
    
    # Symptoms (binary)
    has_diabetes = np.random.binomial(1, 0.3, n_samples)
    has_jaundice = np.random.binomial(1, 0.25, n_samples)
    has_weight_loss = np.random.binomial(1, 0.4, n_samples)
    has_abdominal_pain = np.random.binomial(1, 0.5, n_samples)
    has_nausea = np.random.binomial(1, 0.2, n_samples)
    
    # Combine into DataFrame
    data = {
        'patient_id': [f'PDAC_{i:05d}' for i in range(n_samples)],
        'age': age,
        'CA19-9': ca199,
        'CEA': cea,
        'bilirubin_total': bilirubin,
        'ALT': alt,
        'AST': ast,
        'ALP': alp,
        'glucose': glucose,
        'albumin': albumin,
        'hemoglobin': hemoglobin,
        'has_diabetes': has_diabetes,
        'has_jaundice': has_jaundice,
        'has_weight_loss': has_weight_loss,
        'has_abdominal_pain': has_abdominal_pain,
        'has_nausea': has_nausea,
        'CA125': np.random.lognormal(3, 1.2, n_samples),
        'CRP': np.random.exponential(20, n_samples),
        'creatinine': np.random.lognormal(0.8, 0.3, n_samples),
        'platelets': np.random.normal(250, 80, n_samples)
    }
    
    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print("ClinDiff-PDAC Training Example")
    print("=" * 60)
    
    # 1. Create sample data
    print("\n1. Creating sample data...")
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = data_dir / "sample_pancreatic_data.csv"
    df = create_sample_data(n_samples=1000, n_features=20)
    df.to_csv(data_path, index=False)
    print(f"   Data saved to: {data_path}")
    print(f"   Shape: {df.shape}")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    feature_names = [
        'age', 'CA19-9', 'CEA', 'bilirubin_total', 'ALT', 'AST',
        'ALP', 'glucose', 'albumin', 'hemoglobin', 'has_diabetes',
        'has_jaundice', 'has_weight_loss', 'has_abdominal_pain',
        'has_nausea', 'CA125', 'CRP', 'creatinine', 'platelets'
    ]
    
    categorical_features = ['has_diabetes', 'has_jaundice', 
                           'has_weight_loss', 'has_abdominal_pain', 'has_nausea']
    
    preprocessor = DataPreprocessor(
        feature_names=feature_names,
        categorical_features=categorical_features,
        normalization_method='standard'
    )
    preprocessor.fit(df)
    
    # Save preprocessor
    preprocessor_path = data_dir.parent.parent / "configs" / "preprocessor.json"
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(str(preprocessor_path))
    print(f"   Preprocessor saved to: {preprocessor_path}")
    
    # 3. Create dataset
    print("\n3. Creating dataset...")
    dataset = PDACDataset(
        data_path=str(data_path),
        preprocessor=preprocessor
    )
    print(f"   Dataset size: {len(dataset)}")
    
    # 4. Create model
    print("\n4. Creating model...")
    data_dim = len(feature_names)
    
    model = KnowledgeGuidedDiffusion(
        data_dim=data_dim,
        kg_dim=256,
        hidden_dims=[512, 512, 256],
        num_timesteps=1000
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Training configuration
    print("\n5. Setting up training...")
    config = TrainingConfig(
        data_dim=data_dim,
        kg_dim=256,
        hidden_dims=[512, 512, 256],
        num_timesteps=1000,
        batch_size=64,
        num_epochs=10,  # Few epochs for demo
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_interval=10,
        save_interval=5
    )
    
    print(f"   Device: {config.device}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    
    # 6. Train (simplified - using same data for train/val for demo)
    print("\n6. Starting training...")
    from clindiff_pdac import EMRDataLoader
    
    train_loader = EMRDataLoader(dataset, batch_size=64, shuffle=True)
    
    trainer = ClinDiffTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=None  # Skip validation for demo
    )
    
    history = trainer.train()
    
    # 7. Save final model
    print("\n7. Saving model...")
    final_path = Path("checkpoints") / "final_model.pt"
    final_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'feature_names': feature_names
    }, final_path)
    print(f"   Model saved to: {final_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, preprocessor


if __name__ == "__main__":
    main()

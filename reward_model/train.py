"""
Main training script for reward model.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from reward_model.config.training_config import TrainingConfig
from reward_model.src.trainer import RewardModelTrainer


def main():
    """Main training function."""
    # Set environment variables for memory optimization
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Create training configuration
    config = TrainingConfig()
    
    # Print configuration
    print("Training Configuration:")
    print(f"Base model: {config.base_model_name}")
    print(f"Max samples: {config.max_samples}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Max length: {config.max_length}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    # Initialize trainer
    trainer = RewardModelTrainer(config)
    
    try:
        # Train model
        reward_trainer, model, tokenizer = trainer.train()
        
        # Evaluate model
        print("\nEvaluating trained model:")
        evaluation_results = trainer.evaluate_model()
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

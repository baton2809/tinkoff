"""
Training utilities for reward model.
"""

import os
from typing import Tuple, Optional
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

from .model_manager import RewardModelManager
from .data_loader import RewardDataLoader
from ..config.training_config import TrainingConfig


class RewardModelTrainer:
    """Handles reward model training process."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model_manager = RewardModelManager()
        self.data_loader = RewardDataLoader(config.data_dir)
        
    def setup_training_args(self) -> RewardConfig:
        """
        Setup training arguments from configuration.
        
        Returns:
            RewardConfig object with training parameters
        """
        return RewardConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_length=self.config.max_length,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            disable_dropout=self.config.disable_dropout,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_drop_last=self.config.dataloader_drop_last,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to=self.config.report_to,
            dataloader_num_workers=self.config.dataloader_num_workers,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
        )
    
    def train(self) -> Tuple[RewardTrainer, AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Execute the complete training process.
        
        Returns:
            Tuple of (trainer, model, tokenizer)
            
        Raises:
            RuntimeError: If MPS memory error occurs
            FileNotFoundError: If data directory not found
        """
        print("Starting reward model training")
        
        try:
            # Clear memory at start
            self.model_manager.clear_memory()
            
            # Load model and tokenizer
            print(f"Loading base model: {self.config.base_model_name}")
            model, tokenizer = self.model_manager.load_model_and_tokenizer(
                self.config.base_model_name
            )
            
            print(f"Model parameters: {model.num_parameters():,}")
            
            # Load and prepare datasets
            print("Loading datasets")
            train_dataset, val_dataset = self.data_loader.load_datasets(
                max_samples=self.config.max_samples
            )
            
            print(f"Train samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            
            # Prepare datasets for training
            print("Preparing datasets for training")
            train_prepared = self.data_loader.prepare_for_training(
                train_dataset, self.config.max_length
            )
            val_prepared = self.data_loader.prepare_for_training(
                val_dataset, self.config.max_length
            )
            
            # Clear memory after data preparation
            self.model_manager.clear_memory()
            
            # Setup training arguments
            training_args = self.setup_training_args()
            
            # Create trainer
            print("Creating RewardTrainer")
            trainer = RewardTrainer(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                train_dataset=train_prepared,
                eval_dataset=val_prepared,
            )
            
            # Clear memory before training
            self.model_manager.clear_memory()
            
            # Start training
            print("Starting training process")
            trainer.train()
            
            # Save model
            print("Saving trained model")
            self.model_manager.save_model(model, tokenizer, self.config.output_dir)
            
            print(f"Training completed. Model saved to: {self.config.output_dir}")
            
            return trainer, model, tokenizer
            
        except RuntimeError as e:
            if "MPS backend out of memory" in str(e):
                print("MPS memory error occurred!")
                print("Suggestions:")
                print(f"1. Reduce max_samples (current: {self.config.max_samples})")
                print(f"2. Reduce max_length (current: {self.config.max_length})")
                print(f"3. Reduce batch_size (current: {self.config.per_device_train_batch_size})")
                print("4. Use CPU instead of MPS")
                print("5. Close other applications to free memory")
            raise
    
    def evaluate_model(self, model_path: Optional[str] = None) -> dict:
        """
        Evaluate trained model with test examples.
        
        Args:
            model_path: Path to trained model (uses config.output_dir if None)
            
        Returns:
            Dictionary with evaluation results
        """
        if model_path is None:
            model_path = self.config.output_dir
        
        print(f"Loading model for evaluation: {model_path}")
        model, tokenizer = self.model_manager.load_trained_model(model_path)
        
        # Test examples
        test_cases = [
            {
                "prompt": "How to cook pasta?",
                "good_response": "Boil water, add salt, cook pasta according to package instructions.",
                "bad_response": "I don't know, try searching online."
            },
            {
                "prompt": "Explain machine learning",
                "good_response": "Machine learning is a subset of AI that enables computers to learn from data.",
                "bad_response": "It's complicated stuff with computers."
            }
        ]
        
        results = []
        for case in test_cases:
            comparison = self.model_manager.compare_responses(
                case["prompt"], 
                case["good_response"], 
                case["bad_response"],
                self.config.max_length
            )
            results.append(comparison)
            
            print(f"\nTest case: {case['prompt']}")
            print(f"Good response score: {comparison['score_a']:.4f}")
            print(f"Bad response score: {comparison['score_b']:.4f}")
            print(f"Difference: {comparison['difference']:.4f}")
            print(f"Correct preference: {'Yes' if comparison['preferred'] == 'A' else 'No'}")
        
        return results

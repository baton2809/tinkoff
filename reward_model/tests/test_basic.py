"""
Basic tests for reward model components.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch

from reward_model.config.training_config import TrainingConfig
from reward_model.src.data_loader import RewardDataLoader
from reward_model.src.model_manager import RewardModelManager


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        self.assertEqual(config.base_model_name, "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(config.num_labels, 1)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.max_samples, 500)
        self.assertEqual(config.max_length, 128)
        self.assertEqual(config.per_device_train_batch_size, 1)


class TestRewardDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def test_init(self):
        """Test data loader initialization."""
        data_dir = "test_data"
        loader = RewardDataLoader(data_dir)
        self.assertEqual(loader.data_dir, data_dir)
    
    def test_missing_data_directory(self):
        """Test error handling for missing data directory."""
        loader = RewardDataLoader("nonexistent_dir")
        
        with self.assertRaises(FileNotFoundError):
            loader.load_datasets()


class TestRewardModelManager(unittest.TestCase):
    """Test model management functionality."""
    
    def test_init(self):
        """Test model manager initialization."""
        manager = RewardModelManager()
        self.assertIsNone(manager.model)
        self.assertIsNone(manager.tokenizer)
    
    def test_clear_memory(self):
        """Test memory clearing function."""
        manager = RewardModelManager()
        # Should not raise any errors
        manager.clear_memory()
    
    def test_get_reward_score_without_model(self):
        """Test error handling when model is not loaded."""
        manager = RewardModelManager()
        
        with self.assertRaises(RuntimeError):
            manager.get_reward_score("test text")


if __name__ == "__main__":
    unittest.main()

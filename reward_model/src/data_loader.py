"""
Data loading and preprocessing utilities for reward model training.
"""

import os
from typing import Tuple, Optional
from datasets import Dataset, load_from_disk


class RewardDataLoader:
    """Handles loading and preprocessing of reward model training data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing processed dataset
        """
        self.data_dir = data_dir
        
    def load_datasets(self, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """
        Load train and validation datasets.
        
        Args:
            max_samples: Maximum number of training samples to use
            
        Returns:
            Tuple of (train_dataset, val_dataset)
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory {self.data_dir} not found. "
                "Please run data preprocessing first."
            )
        
        # Load datasets
        train_dataset = load_from_disk(os.path.join(self.data_dir, "train"))
        val_dataset = load_from_disk(os.path.join(self.data_dir, "validation"))
        
        # Validate required columns
        required_columns = ["prompt", "chosen", "rejected"]
        missing_columns = [col for col in required_columns if col not in train_dataset.column_names]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Limit dataset size if specified
        if max_samples is not None:
            if len(train_dataset) > max_samples:
                train_dataset = train_dataset.select(range(max_samples))
            
            val_max = min(max_samples // 4, len(val_dataset))
            if len(val_dataset) > val_max:
                val_dataset = val_dataset.select(range(val_max))
        
        return train_dataset, val_dataset
    
    def prepare_for_training(self, dataset: Dataset, max_length: int = 128) -> Dataset:
        """
        Prepare dataset for reward model training.
        
        Args:
            dataset: Input dataset
            max_length: Maximum text length for truncation
            
        Returns:
            Prepared dataset with 'chosen' and 'rejected' columns
        """
        def combine_prompt_and_response(examples):
            """Combine prompts with responses and truncate if needed."""
            chosen_texts = []
            rejected_texts = []
            
            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                chosen = examples["chosen"][i]
                rejected = examples["rejected"][i]
                
                # Truncate long texts to prevent memory issues
                if len(prompt) > 200:
                    prompt = prompt[:200] + "..."
                if len(chosen) > 200:
                    chosen = chosen[:200] + "..."
                if len(rejected) > 200:
                    rejected = rejected[:200] + "..."
                
                # Combine prompt with responses
                chosen_text = f"{prompt}\n{chosen}"
                rejected_text = f"{prompt}\n{rejected}"
                
                chosen_texts.append(chosen_text)
                rejected_texts.append(rejected_text)
            
            return {
                "chosen": chosen_texts,
                "rejected": rejected_texts,
            }
        
        # Process dataset
        prepared_dataset = dataset.map(
            combine_prompt_and_response,
            batched=True,
            batch_size=100,
            desc="Preparing dataset for training"
        )
        
        # Keep only required columns
        columns_to_keep = ["chosen", "rejected"]
        columns_to_remove = [
            col for col in prepared_dataset.column_names 
            if col not in columns_to_keep
        ]
        
        prepared_dataset = prepared_dataset.remove_columns(columns_to_remove)
        
        return prepared_dataset

import os
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.device_utils import clear_memory_before


class RewardModelManager:
    """Manages reward model loading, training, and inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self, model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load model and tokenizer for training.
        
        Args:
            model_path: path of the model
            
        Returns:
            Tuple of (model, tokenizer)
        """

        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def save_model(self, model: AutoModelForSequenceClassification,
                   tokenizer: AutoTokenizer, output_dir: str) -> None:
        """
        Save trained model and tokenizer.
        
        Args:
            model: Trained model to save
            tokenizer: Tokenizer to save
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        metadata = {
            "model_type": "reward_model",
            "base_model": model.config.name_or_path,
            "num_parameters": model.num_parameters(),
            "torch_dtype": str(model.dtype),
        }

        import json
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @clear_memory_before
    def load_trained_model(self, model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load a trained reward model for inference.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def get_reward_score(self, text: str, max_length: int = 128) -> float:
        """
        Get reward score for a given text.
        
        Args:
            text: Input text to score
            max_length: Maximum sequence length
            
        Returns:
            Reward score as float
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_trained_model first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            reward_score = outputs.logits.item()

        return reward_score

    def compare_responses(self, prompt: str, response_a: str, response_b: str,
                          max_length: int = 128) -> dict:
        """
        Compare two responses for a given prompt.
        
        Args:
            prompt: Input prompt
            response_a: First response
            response_b: Second response
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with comparison results
        """
        text_a = f"{prompt}\n{response_a}"
        text_b = f"{prompt}\n{response_b}"

        score_a = self.get_reward_score(text_a, max_length)
        score_b = self.get_reward_score(text_b, max_length)

        return {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "score_a": score_a,
            "score_b": score_b,
            "difference": score_a - score_b,
            "preferred": "A" if score_a > score_b else "B"
        }

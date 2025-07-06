from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration class for reward model training."""

    # Model configuration
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    num_labels: int = 1

    # Training parameters
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1  # increase to speed up / decrease to avoid OOM
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 128  # default value, will be overridden from metadata.json if available
    split_ratio: int = 19  # ratio for train/validation split 95/5

    # Memory optimization
    max_samples: Optional[int] = 500
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0

    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 400
    logging_steps: int = 100
    save_total_limit: int = 2

    # Optimization
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    disable_dropout: bool = True

    remove_unused_columns: bool = False
    dataloader_drop_last: bool = True
    load_best_model_at_end: bool = False
    report_to: Optional[str] = None

    model_path: str = "reward_model_output"
    data_dir: str = "processed_dataset"
    output_dir: str = "trained_model"

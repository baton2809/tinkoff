# Reward Model Training

Clean, production-ready implementation of reward model training for RLHF (Reinforcement Learning from Human Feedback).

## Project Structure

```
reward_model/
├── config/
│   └── training_config.py    # Training configuration
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading utilities
│   ├── model_manager.py     # Model management
│   └── trainer.py           # Training logic
├── models/
│   └── trained_model/       # Saved model directory
├── tests/                   # Unit tests
├── train.py                 # Main training script
├── inference.py             # Inference script
└── README.md               # This file
```

## Features

- Memory-optimized training for Apple Silicon (MPS)
- Clean, modular architecture
- Comprehensive error handling
- Model persistence with metadata
- Interactive inference interface
- Configurable training parameters

## Quick Start

### 1. Training

```bash
# Train with default configuration
cd reward_model
python train.py
```

### 2. Inference

```bash
# Basic test
python inference.py

# Interactive mode
python inference.py --interactive

# Score single text
python inference.py --text "Your text here"

# Compare responses
python inference.py --prompt "Question" --response_a "Good answer" --response_b "Bad answer"
```

## Configuration

Edit `config/training_config.py` to customize training parameters:

```python
@dataclass
class TrainingConfig:
    # Model
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # Training
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    max_samples: int = 500
    max_length: int = 128
    
    # Memory optimization
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
```

## Memory Optimization

The implementation includes several memory optimization techniques:

- Automatic memory clearing between steps
- Reduced batch sizes for MPS compatibility
- Text truncation to prevent memory overflow
- Environment variable configuration for PyTorch
- Fallback to CPU when MPS fails

## Model Persistence

Trained models are saved with:
- Model weights and configuration
- Tokenizer configuration
- Training metadata
- Model performance metrics

## API Usage

```python
from reward_model.src.model_manager import RewardModelManager

# Load trained model
manager = RewardModelManager()
model, tokenizer = manager.load_trained_model("models/trained_model")

# Get reward score
score = manager.get_reward_score("Your text here")

# Compare responses
result = manager.compare_responses(
    prompt="How to cook pasta?",
    response_a="Boil water, add salt, cook pasta according to instructions.",
    response_b="I don't know."
)
```

python inference.py --prompt "How to cook pasta?" --response_a "Boil water, add salt, cook pasta according to instructions." --response_b "I don't know."

## Requirements

- Python 3.8+
- PyTorch with MPS support
- Transformers
- TRL (Transformer Reinforcement Learning)
- Datasets

## Troubleshooting

### MPS Memory Errors

If you encounter MPS memory errors:

1. Reduce `max_samples` in configuration
2. Reduce `max_length` parameter
3. Set `per_device_train_batch_size=1`
4. Close other applications
5. Use CPU fallback: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

### Data Loading Issues

Ensure your dataset is in the correct format with columns:
- `prompt`: Input prompts
- `chosen`: Preferred responses
- `rejected`: Non-preferred responses

## Development

### Running Tests

```bash
cd reward_model
python -m pytest tests/
```

### Code Style

The codebase follows:
- PEP 8 style guidelines
- Type hints for all functions
- Comprehensive docstrings
- Clean separation of concerns

## License

This project is part of the RLHF training pipeline.

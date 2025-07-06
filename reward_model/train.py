import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from reward_model.config.training_config import TrainingConfig
from reward_model.src.trainer import RewardModelTrainer


def override_config(data_dir: str = "processed_dataset"):
    config = TrainingConfig()

    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            config.max_length = metadata['max_length']
            config.batch_size = metadata['batch_size']

    return config


if __name__ == "__main__":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['WANDB_SILENT'] = 'True'

    config = override_config()

    print("Конфигурация:")
    print(f"Base model: {config.base_model_name}")
    print(f"Max samples: {config.max_samples}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Max length: {config.max_length}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Output directory: {config.output_dir}\n")

    trainer = RewardModelTrainer(config)

    try:
        print("Обучение Reward Model...")
        trainer.train()

        print("Оценка модели:")
        trainer.evaluate_model()

        print("\nОбучение завершено")
        print(f"Модель сохранена: {config.output_dir}")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

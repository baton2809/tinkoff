import os
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

from utils.device_utils import get_model_loading_config, clear_memory_before
from .data_loader import RewardDataLoader
from .model_manager import RewardModelManager
from ..config.training_config import TrainingConfig


class RewardModelTrainer:
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model_manager = RewardModelManager()
        self.data_loader = RewardDataLoader(config.data_dir)
        self.model = None
        self.tokenizer = None

    @clear_memory_before
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

    def train(self):
        model_path = self.config.output_dir

        if os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Обнаружена обученная модель в {model_path}, загружаем...")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Модель успешно загружена из {model_path}")
            return

        try:
            print(f"Загружаем SFT: {self.config.base_model_name}")
            model_config = get_model_loading_config()

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.base_model_name,
                num_labels=1,
                **model_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

            print(f"Количество параметров модели: {self.model.num_parameters():,}")

            print(f"Загружаем датасет из {self.config.data_dir}...")
            train_dataset, val_dataset = self.data_loader.load_datasets(
                split_ratio=self.config.split_ratio,
                max_samples=self.config.max_samples
            )

            training_args = self.setup_training_args()

            print("Создаем RewardTrainer...")
            trainer = RewardTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            print("Начинаем процесс обучения...")
            trainer.train()

            print("Сохраняем обученную модель...")
            self.model_manager.save_model(self.model, self.tokenizer, self.config.output_dir)

            print(f"Обучение завершено. Модель сохранена в: {self.config.output_dir}")

        except RuntimeError as e:
            if "MPS Out of Memory" in str(e):
                print("\nРекомендации:")
                print(f"1. Уменьшить max_samples (сейчас: {self.config.max_samples})")
                print(f"2. Уменьшить max_length (сейчас: {self.config.max_length})")
                print(f"3. Уменьшить batch_size (сейчас: {self.config.per_device_train_batch_size})")
                print("4. Закрыть другие приложения для освобождения памяти")
                print("5. Перезапустить Python для очистки памяти")
            raise

    def evaluate_model(self, model_path: Optional[str] = None) -> None:
        """
        Evaluate trained model with test examples.
        
        Args:
            model_path: Path to trained model (uses config.output_dir if None)
            
        Returns:
            Dictionary with evaluation results
        """
        if model_path is None:
            model_path = self.config.output_dir

        print(f"Загружаем модель для оценки: {model_path}")
        self.model_manager.load_trained_model(model_path)

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

            print(f"\nТестовый пример: {case['prompt']}")
            print(f"Принятый ответ из dataset: {case['good_response']}")
            print(f"Отвергнутый ответ из dataset: {case['bad_response']}")
            print(f"Оценка хорошего ответа: {comparison['score_a']:.4f}")
            print(f"Оценка плохого ответа: {comparison['score_b']:.4f}")
            print(f"Разница: {comparison['difference']:.4f}")
            print(f"Правильное предпочтение: {'Да' if comparison['preferred'] == 'A' else 'Нет'}")

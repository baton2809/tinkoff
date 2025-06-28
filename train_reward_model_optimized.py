"""
Оптимизированное обучение Reward Model с управлением памяти для MPS
Включает техники экономии памяти для Apple Silicon устройств
"""

import torch
import gc
import os
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
)
from trl import RewardTrainer, RewardConfig
from datasets import load_from_disk
from typing import Dict, Any

def clear_memory():
    """Очищает память GPU/MPS"""
    if torch.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_reward_model_and_tokenizer(model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
    """
    Загружает модель для sequence classification (reward model) и токенизатор
    с оптимизациями для экономии памяти
    """
    print(f"Загружаем reward model из: {model_name}")
    
    # Очищаем память перед загрузкой
    clear_memory()
    
    # Используем float32 для MPS, так как float16 может вызывать проблемы
    if torch.mps.is_available():
        torch_dtype = torch.float32
        device_map = None  # Отключаем device_map для MPS
        print("⚠ Используется torch.float32 для MPS устройства")
        print("⚠ Device_map отключен для лучшей совместимости с MPS")
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Для reward model нужен один выход (reward score)
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,  # Экономия памяти при загрузке
    )
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Устанавливаем pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Обновляем конфигурацию модели для pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"✓ Reward model загружена: {model.config.name_or_path}")
    print(f"✓ Количество параметров: {model.num_parameters():,}")
    print(f"✓ Токенизатор настроен, pad_token: {tokenizer.pad_token}")
    
    return model, tokenizer

def load_processed_dataset(data_dir: str = "processed_helpsteer", max_samples: int = None):
    """
    Загружает обработанный датасет с парами chosen/rejected
    Добавлена возможность ограничить количество примеров для экономии памяти
    """
    print(f"Загружаем обработанный датасет из: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Директория {data_dir} не найдена. Сначала запустите load_helpsteer_dataset.py")
    
    # Загружаем train и validation датасеты
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(data_dir, "validation"))
    
    # Ограничиваем размер датасета для экономии памяти
    if max_samples is not None:
        if len(train_dataset) > max_samples:
            train_dataset = train_dataset.select(range(max_samples))
            print(f"⚠ Train dataset ограничен до {max_samples} примеров")
        
        val_max = min(max_samples // 4, len(val_dataset))  # 25% от train для validation
        if len(val_dataset) > val_max:
            val_dataset = val_dataset.select(range(val_max))
            print(f"⚠ Validation dataset ограничен до {val_max} примеров")
    
    print(f"✓ Train dataset: {len(train_dataset)} примеров")
    print(f"✓ Validation dataset: {len(val_dataset)} примеров")
    print(f"✓ Колонки: {train_dataset.column_names}")
    
    # Проверяем наличие необходимых колонок
    required_columns = ["prompt", "chosen", "rejected"]
    missing_columns = [col for col in required_columns if col not in train_dataset.column_names]
    
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
    
    return train_dataset, val_dataset

def prepare_dataset_for_reward_training(dataset, tokenizer, max_length: int = 128):
    """
    Подготавливает датасет для обучения reward model
    Уменьшена максимальная длина для экономии памяти
    """
    print(f"Подготавливаем датасет для reward training (max_length={max_length})...")
    
    def combine_prompt_and_response(examples):
        """
        Комбинирует prompt с chosen и rejected ответами
        Обрезает слишком длинные тексты для экономии памяти
        """
        chosen_texts = []
        rejected_texts = []
        
        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            chosen = examples["chosen"][i]
            rejected = examples["rejected"][i]
            
            # Обрезаем слишком длинные промпты и ответы
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            if len(chosen) > 200:
                chosen = chosen[:200] + "..."
            if len(rejected) > 200:
                rejected = rejected[:200] + "..."
            
            # Создаем полные тексты для chosen и rejected
            chosen_text = f"{prompt}\n{chosen}"
            rejected_text = f"{prompt}\n{rejected}"
            
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)
        
        return {
            "chosen": chosen_texts,
            "rejected": rejected_texts,
        }
    
    # Комбинируем prompt с ответами
    prepared_dataset = dataset.map(
        combine_prompt_and_response,
        batched=True,
        batch_size=100,  # Уменьшенный batch size для экономии памяти
        desc="Combining prompts with responses"
    )
    
    # Удаляем ненужные колонки
    columns_to_keep = ["chosen", "rejected"]
    columns_to_remove = [col for col in prepared_dataset.column_names 
                        if col not in columns_to_keep]
    
    prepared_dataset = prepared_dataset.remove_columns(columns_to_remove)
    
    print(f"✓ Датасет подготовлен для reward training")
    print(f"✓ Колонки: {prepared_dataset.column_names}")
    
    return prepared_dataset

def setup_training_arguments(
    output_dir: str = "./reward_model_output",
    learning_rate: float = 5e-5,
    num_train_epochs: int = 1,
    max_length: int = 128,  # Уменьшено с 256 до 128
    batch_size: int = 1,    # Уменьшено с 2 до 1
) -> RewardConfig:
    """
    Настраивает аргументы для обучения с оптимизациями для MPS
    """
    print("Настраиваем параметры обучения для экономии памяти...")
    
    # Отключаем FP16 и BF16 для MPS
    fp16 = False
    bf16 = False
    if torch.mps.is_available():
        print("⚠ FP16 и BF16 отключены для MPS устройства")
    
    training_args = RewardConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        bf16=bf16,
        max_length=max_length,
        disable_dropout=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Увеличено для компенсации меньшего batch_size
        eval_strategy="steps",
        eval_steps=200,  # Увеличено для экономии памяти
        save_strategy="steps",
        save_steps=400,
        logging_steps=100,
        warmup_steps=50,
        max_grad_norm=1.0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=None,
        dataloader_num_workers=0,  # Отключаем многопоточность для экономии памяти
        save_total_limit=2,  # Ограничиваем количество сохраненных чекпоинтов
        load_best_model_at_end=False,  # Отключаем для экономии памяти
    )
    
    print(f"✓ Learning rate: {learning_rate}")
    print(f"✓ Epochs: {num_train_epochs}")
    print(f"✓ Max length: {max_length}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"✓ Effective batch size: {batch_size * training_args.gradient_accumulation_steps}")
    
    return training_args

def train_reward_model(max_samples: int = 1000):
    """
    Основная функция для обучения reward model с ограничениями памяти
    """
    print("=== Обучение Reward Model (Memory Optimized) ===\n")
    
    try:
        # Очищаем память в начале
        clear_memory()
        
        # 1. Загружаем модель и токенизатор
        model, tokenizer = load_reward_model_and_tokenizer()
        
        # 2. Загружаем датасет с ограничением
        train_dataset, val_dataset = load_processed_dataset(max_samples=max_samples)
        
        # 3. Подготавливаем датасет для reward training
        train_dataset_prepared = prepare_dataset_for_reward_training(
            train_dataset, tokenizer, max_length=128
        )
        val_dataset_prepared = prepare_dataset_for_reward_training(
            val_dataset, tokenizer, max_length=128
        )
        
        # Очищаем память после подготовки данных
        clear_memory()
        
        # 4. Настраиваем параметры обучения
        training_args = setup_training_arguments(
            learning_rate=5e-5,
            num_train_epochs=1,
            max_length=128,
            batch_size=1,
        )
        
        # 5. Создаем RewardTrainer
        print("Создаем RewardTrainer...")
        trainer = RewardTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset_prepared,
            eval_dataset=val_dataset_prepared,
        )
        
        print("✓ RewardTrainer создан")
        
        # Очищаем память перед обучением
        clear_memory()
        
        # 6. Запускаем обучение
        print("\n=== Начинаем обучение ===")
        trainer.train()
        
        # 7. Сохраняем обученную модель
        print("\n=== Сохраняем модель ===")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        print(f"✓ Модель сохранена в: {training_args.output_dir}")
        
        # 8. Показываем финальную статистику
        print("\n=== Обучение завершено ===")
        print(f"✓ Модель обучена на {len(train_dataset_prepared)} примерах")
        print(f"✓ Валидация на {len(val_dataset_prepared)} примерах")
        print(f"✓ Параметры: lr={training_args.learning_rate}, epochs={training_args.num_train_epochs}")
        print(f"✓ Результат сохранен в: {training_args.output_dir}")
        
        return trainer, model, tokenizer
        
    except RuntimeError as e:
        if "MPS backend out of memory" in str(e):
            print("\n❌ Ошибка нехватки памяти MPS!")
            print("Попробуйте следующие решения:")
            print("1. Уменьшите max_samples (сейчас: {})".format(max_samples))
            print("2. Уменьшите max_length (сейчас: 128)")
            print("3. Установите batch_size=1")
            print("4. Используйте CPU вместо MPS")
            print("5. Закройте другие приложения для освобождения памяти")
            print("\nДля использования CPU добавьте в начало скрипта:")
            print("os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'")
            print("или")
            print("os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'")
        raise

def test_reward_model(model_path: str = "./reward_model_output"):
    """
    Тестирует обученную reward model
    """
    print(f"\n=== Тестирование Reward Model ===")
    print(f"Загружаем модель из: {model_path}")
    
    # Очищаем память перед загрузкой
    clear_memory()
    
    # Загружаем обученную модель
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Тестовый пример
    prompt = "Как приготовить пасту?"
    chosen = "Вскипятите воду, добавьте соль, варите пасту согласно инструкции."
    rejected = "Не знаю, попробуйте поискать в интернете."
    
    # Подготавливаем тексты
    chosen_text = f"{prompt}\n{chosen}"
    rejected_text = f"{prompt}\n{rejected}"
    
    # Токенизируем
    chosen_inputs = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=128)
    rejected_inputs = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=128)
    
    # Получаем reward scores
    with torch.no_grad():
        chosen_reward = model(**chosen_inputs).logits.item()
        rejected_reward = model(**rejected_inputs).logits.item()
    
    print(f"\nТестовый пример:")
    print(f"Prompt: {prompt}")
    print(f"Chosen: {chosen}")
    print(f"Rejected: {rejected}")
    print(f"\nReward scores:")
    print(f"Chosen reward: {chosen_reward:.4f}")
    print(f"Rejected reward: {rejected_reward:.4f}")
    print(f"Разница: {chosen_reward - rejected_reward:.4f}")
    
    if chosen_reward > rejected_reward:
        print("✓ Модель правильно предпочитает chosen ответ!")
    else:
        print("⚠ Модель предпочитает rejected ответ")

if __name__ == "__main__":
    # Опциональные переменные окружения для управления памятью MPS
    # Раскомментируйте при необходимости:
    
    # Отключить верхний лимит памяти MPS (ОСТОРОЖНО!)
    # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Включить fallback на CPU при ошибках MPS
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Проверяем доступность устройства
    if torch.mps.is_available():
        device = torch.device("mps")
        print(f"Используется устройство: {device}")
        print("⚠ Для MPS применены оптимизации памяти")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Используется устройство: {device}")
    else:
        device = torch.device("cpu")
        print(f"Используется устройство: {device}")
    
    try:
        # Обучаем reward model с ограничением примеров
        trainer, model, tokenizer = train_reward_model(max_samples=500)  # Начинаем с малого
        
        # Тестируем результат
        test_reward_model()
        
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n=== Рекомендации по решению проблем ===")
        print("1. Попробуйте уменьшить max_samples до 100-200")
        print("2. Используйте CPU вместо MPS:")
        print("   export PYTORCH_ENABLE_MPS_FALLBACK=1")
        print("3. Закройте другие приложения для освобождения памяти")
        print("4. Перезапустите Python для очистки памяти")

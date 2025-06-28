"""
Пример использования обработанного датасета HelpSteer2_binarized
Демонстрация загрузки и использования для обучения
"""

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def load_processed_data():
    """
    Загружает обработанные датасеты
    """
    print("=== Загрузка обработанных датасетов ===")
    
    # Загружаем датасеты
    train_dataset = load_from_disk("processed_helpsteer/train")
    val_dataset = load_from_disk("processed_helpsteer/validation")
    
    # Загружаем метаданные
    with open("processed_helpsteer/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"✓ Train dataset: {len(train_dataset)} примеров")
    print(f"✓ Validation dataset: {len(val_dataset)} примеров")
    print(f"✓ Максимальная длина токенов: {metadata['max_length']}")
    print(f"✓ Токенизатор: {metadata['tokenizer']}")
    
    return train_dataset, val_dataset, metadata

def demonstrate_data_access(train_dataset, val_dataset):
    """
    Демонстрирует доступ к данным
    """
    print("\n=== Демонстрация доступа к данным ===")
    
    # Показываем структуру данных
    print(f"Колонки в датасете: {train_dataset.column_names}")
    
    # Берем первый пример
    example = train_dataset[0]
    print(f"\nПример данных:")
    print(f"- Prompt: {example['prompt'][:100]}...")
    print(f"- Chosen rating: {example['chosen_rating']}")
    print(f"- Rejected rating: {example['rejected_rating']}")
    print(f"- Input IDs length: {len(example['input_ids'])}")
    print(f"- Attention mask length: {len(example['attention_mask'])}")
    
    # Показываем как получить батч данных
    batch = train_dataset.select(range(4))  # Первые 4 примера
    print(f"\nБатч из {len(batch)} примеров:")
    print(f"- Prompts: {[p[:50] + '...' for p in batch['prompt']]}")
    print(f"- Chosen ratings: {batch['chosen_rating']}")
    print(f"- Rejected ratings: {batch['rejected_rating']}")

def demonstrate_dataloader_usage(train_dataset, val_dataset):
    """
    Демонстрирует использование с DataLoader
    """
    print("\n=== Демонстрация использования с DataLoader ===")
    
    from torch.utils.data import DataLoader
    
    # Функция для преобразования в тензоры
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        chosen_ratings = torch.tensor([item['chosen_rating'] for item in batch])
        rejected_ratings = torch.tensor([item['rejected_rating'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'chosen_ratings': chosen_ratings,
            'rejected_ratings': rejected_ratings,
            'prompts': [item['prompt'] for item in batch]
        }
    
    # Создаем DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Демонстрируем использование
    print(f"Train DataLoader: {len(train_dataloader)} батчей")
    print(f"Validation DataLoader: {len(val_dataloader)} батчей")
    
    # Берем первый батч
    batch = next(iter(train_dataloader))
    print(f"\nПример батча:")
    print(f"- Input IDs shape: {batch['input_ids'].shape}")
    print(f"- Attention mask shape: {batch['attention_mask'].shape}")
    print(f"- Chosen ratings: {batch['chosen_ratings']}")
    print(f"- Rejected ratings: {batch['rejected_ratings']}")
    print(f"- Prompts: {[p[:30] + '...' for p in batch['prompts']]}")

def demonstrate_training_setup():
    """
    Демонстрирует настройку для обучения
    """
    print("\n=== Демонстрация настройки для обучения ===")
    
    # Проверяем устройство
    if torch.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Используется устройство: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Используется устройство: {device}")
    else:
        device = torch.device("cpu")
        print(f"✓ Используется устройство: {device}")
    
    print(f"\nПример настройки для обучения:")
    print(f"""
# Пример кода для SFT (Supervised Fine-Tuning)
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Маленький batch для экономии памяти
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,  # Для экономии памяти
    dataloader_pin_memory=False,  # Для MPS
    remove_unused_columns=False,
)

# Для DPO (Direct Preference Optimization)
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo_results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=5e-7,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
)
""")

def demonstrate_filtering_examples():
    """
    Демонстрирует фильтрацию примеров по рейтингу
    """
    print("\n=== Демонстрация фильтрации данных ===")
    
    train_dataset, val_dataset, metadata = load_processed_data()
    
    # Фильтруем примеры с высокой разницей в рейтингах
    def filter_high_preference_gap(example):
        gap = example['chosen_rating'] - example['rejected_rating']
        return gap >= 0.5  # Разница в рейтинге минимум 0.5
    
    filtered_train = train_dataset.filter(filter_high_preference_gap)
    filtered_val = val_dataset.filter(filter_high_preference_gap)
    
    print(f"Исходный train dataset: {len(train_dataset)} примеров")
    print(f"Отфильтрованный train dataset: {len(filtered_train)} примеров")
    print(f"Исходный validation dataset: {len(val_dataset)} примеров")
    print(f"Отфильтрованный validation dataset: {len(filtered_val)} примеров")
    
    # Показываем статистику разниц в рейтингах
    gaps = [ex['chosen_rating'] - ex['rejected_rating'] for ex in train_dataset]
    avg_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    max_gap = max(gaps)
    
    print(f"\nСтатистика разниц в рейтингах:")
    print(f"- Средняя разница: {avg_gap:.2f}")
    print(f"- Минимальная разница: {min_gap:.2f}")
    print(f"- Максимальная разница: {max_gap:.2f}")
    
    return filtered_train, filtered_val

def main():
    """
    Основная демонстрационная функция
    """
    print("=== Пример использования обработанного датасета HelpSteer2_binarized ===\n")
    
    # 1. Загружаем данные
    train_dataset, val_dataset, metadata = load_processed_data()
    
    # 2. Демонстрируем доступ к данным
    demonstrate_data_access(train_dataset, val_dataset)
    
    # 3. Демонстрируем использование с DataLoader
    demonstrate_dataloader_usage(train_dataset, val_dataset)
    
    # 4. Демонстрируем настройку для обучения
    demonstrate_training_setup()
    
    # 5. Демонстрируем фильтрацию данных
    filtered_train, filtered_val = demonstrate_filtering_examples()
    
    print(f"\n=== Заключение ===")
    print(f"✓ Датасет успешно загружен и готов к использованию")
    print(f"✓ Поддерживается SFT, DPO и другие методы обучения")
    print(f"✓ Максимальная длина токенов ограничена до 256 для предотвращения OOM")
    print(f"✓ Данные оптимизированы для M1 MacBook с MPS")
    print(f"✓ Доступна фильтрация по качеству предпочтений")

if __name__ == "__main__":
    main()

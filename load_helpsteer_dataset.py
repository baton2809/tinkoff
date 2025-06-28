"""
Загрузка и обработка датасета HelpSteer2_binarized
Разделение на train/validation и ограничение длины токенов
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

def load_and_explore_dataset():
    """
    Загружает датасет и исследует его структуру
    """
    # Список возможных названий датасетов для попытки загрузки
    dataset_candidates = [
        "esfrankel17/HelpSteer2_binarized",
        "esfrankel17/original_HelpSteer2_binarized", 
        "juyoungml/HelpSteer2-binarized",
        "dogtooth/helpsteer2_binarized",
        "sablo/HelpSteer_binarized",
        "KatoHF/helpsteer_binarized",
        "jan-hq/nvidia_helpsteer_binarized"
    ]
    
    dataset = None
    dataset_name = None
    
    for candidate in dataset_candidates:
        print(f"Пробуем загрузить датасет: {candidate}")
        try:
            dataset = load_dataset(candidate)
            dataset_name = candidate
            print(f"✓ Датасет {candidate} успешно загружен!")
            break
        except Exception as e:
            print(f"✗ Ошибка при загрузке {candidate}: {str(e)[:100]}...")
            continue
    
    if dataset is None:
        print("Не удалось загрузить ни один из кандидатов датасетов")
        return None
        
    print(f"\nИспользуем датасет: {dataset_name}")
    print(f"Доступные splits: {list(dataset.keys())}")
    
    # Ищем подходящий split
    target_split = None
    if "average_rating_split" in dataset:
        target_split = "average_rating_split"
    elif "train" in dataset:
        target_split = "train"
        print("Split 'average_rating_split' не найден, используем 'train'")
    else:
        # Берем первый доступный split
        target_split = list(dataset.keys())[0]
        print(f"Используем первый доступный split: '{target_split}'")
    
    split_data = dataset[target_split]
    print(f"\nИнформация о {target_split}:")
    print(f"Количество примеров: {len(split_data)}")
    print(f"Колонки: {split_data.column_names}")
    
    # Показываем первый пример
    print(f"\nПример данных:")
    first_example = split_data[0]
    for key, value in first_example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
            
    return dataset, target_split

def split_dataset(dataset, target_split, test_size=0.2, seed=42):
    """
    Разделяет датасет на train и validation
    """
    print(f"\nРазделяем датасет на train ({int((1-test_size)*100)}%) и validation ({int(test_size*100)}%)...")
    
    # Получаем нужный split
    split_dataset = dataset[target_split]
    
    # Разделяем на train/validation
    train_val_split = split_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    print(f"Train dataset: {len(train_dataset)} примеров")
    print(f"Validation dataset: {len(val_dataset)} примеров")
    
    return train_dataset, val_dataset

def setup_tokenizer(model_name="microsoft/DialoGPT-medium"):
    """
    Настраивает токенизатор
    """
    print(f"\nНастраиваем токенизатор: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Добавляем pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Токенизатор готов. Vocab size: {tokenizer.vocab_size}")
    return tokenizer

def tokenize_function(examples, tokenizer, max_length=256, text_column="prompt"):
    """
    Функция токенизации с ограничением длины
    """
    # Проверяем какие колонки с текстом доступны
    if text_column not in examples:
        # Ищем подходящую колонку с текстом
        text_columns = [col for col in examples.keys() if isinstance(examples[col][0] if examples[col] else "", str)]
        if text_columns:
            text_column = text_columns[0]
            print(f"Используем колонку '{text_column}' для токенизации")
        else:
            raise ValueError("Не найдена подходящая текстовая колонка")
    
    # Токенизируем с ограничением длины
    tokenized = tokenizer(
        examples[text_column],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None  # Возвращаем списки, не тензоры
    )
    
    return tokenized

def process_dataset(train_dataset, val_dataset, tokenizer, max_length=256):
    """
    Обрабатывает датасеты с токенизацией
    """
    print(f"\nОбрабатываем датасеты с максимальной длиной {max_length} токенов...")
    
    # Определяем колонку с текстом
    sample = train_dataset[0]
    text_columns = [col for col in sample.keys() if isinstance(sample[col], str) and len(sample[col]) > 10]
    
    if not text_columns:
        print("Доступные колонки:", list(sample.keys()))
        raise ValueError("Не найдена подходящая текстовая колонка")
    
    text_column = text_columns[0]  # Берем первую подходящую колонку
    print(f"Используем колонку '{text_column}' для токенизации")
    
    # Применяем токенизацию к train dataset
    print("Токенизируем train dataset...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length, text_column),
        batched=True,
        desc="Tokenizing train dataset"
    )
    
    # Применяем токенизацию к validation dataset
    print("Токенизируем validation dataset...")
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length, text_column),
        batched=True,
        desc="Tokenizing validation dataset"
    )
    
    print("Токенизация завершена!")
    print(f"Train dataset: {len(tokenized_train)} примеров")
    print(f"Validation dataset: {len(tokenized_val)} примеров")
    
    return tokenized_train, tokenized_val

def main():
    """
    Основная функция
    """
    print("=== Загрузка и обработка датасета HelpSteer2_binarized ===\n")
    
    # 1. Загружаем и исследуем датасет
    result = load_and_explore_dataset()
    if result is None:
        return None, None
    
    dataset, target_split = result
    
    # 2. Разделяем на train/validation
    train_dataset, val_dataset = split_dataset(dataset, target_split, test_size=0.2, seed=42)
    
    # 3. Настраиваем токенизатор
    tokenizer = setup_tokenizer()
    
    # 4. Обрабатываем датасеты с токенизацией
    tokenized_train, tokenized_val = process_dataset(
        train_dataset, 
        val_dataset, 
        tokenizer, 
        max_length=256
    )
    
    # 5. Сохраняем обработанные датасеты
    from dataset_summary import save_processed_datasets
    output_dir = save_processed_datasets(tokenized_train, tokenized_val)
    
    # 6. Показываем статистику
    print("\n=== Финальная статистика ===")
    print(f"Train dataset: {len(tokenized_train)} примеров")
    print(f"Validation dataset: {len(tokenized_val)} примеров")
    print(f"Максимальная длина токенов: 256")
    print(f"Колонки в обработанном датасете: {tokenized_train.column_names}")
    
    # Показываем пример токенизированных данных
    print(f"\nПример токенизированных данных:")
    example = tokenized_train[0]
    print(f"Input IDs shape: {len(example['input_ids'])}")
    print(f"Attention mask shape: {len(example['attention_mask'])}")
    
    print(f"\n✓ Датасеты сохранены в директории: {output_dir}")
    print(f"✓ Для анализа запустите: python dataset_summary.py")
    
    return tokenized_train, tokenized_val

if __name__ == "__main__":
    # Проверяем доступность устройства
    if torch.mps.is_available():
        device = torch.device("mps")
        print(f"Используется устройство: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Используется устройство: {device}")
    else:
        device = torch.device("cpu")
        print(f"Используется устройство: {device}")
    
    # Запускаем основную функцию
    train_data, val_data = main()

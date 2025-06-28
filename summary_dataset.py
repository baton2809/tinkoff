from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

def save_processed_datasets(train_dataset, val_dataset, output_dir="processed_dataset"):
    """
    Сохраняет обработанные датасеты
    """
    print(f"Сохраняем обработанные датасеты в {output_dir}/...")
    
    # Создаем директорию если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем датасеты
    train_dataset.save_to_disk(f"{output_dir}/train")
    val_dataset.save_to_disk(f"{output_dir}/validation")
    
    print(f"✓ Train dataset сохранен в {output_dir}/train")
    print(f"✓ Validation dataset сохранен в {output_dir}/validation")
    
    # Сохраняем метаданные
    metadata = {
        "dataset_name": "juyoungml/original_HelpSteer2_binarized",
        "split_used": "average_rating",
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "max_length": 128,
        "tokenizer": "microsoft/DialoGPT-medium",
        "columns": train_dataset.column_names,
        "split_ratio": "80/20"
    }
    
    with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Метаданные сохранены в {output_dir}/metadata.json")
    return output_dir

def analyze_dataset_statistics(train_dataset, val_dataset):
    """
    Анализирует статистику датасета
    """
    print("\n=== Детальная статистика датасета ===")
    
    # Общая статистика
    print(f"Общее количество примеров: {len(train_dataset) + len(val_dataset)}")
    print(f"Train: {len(train_dataset)} ({len(train_dataset)/(len(train_dataset) + len(val_dataset))*100:.1f}%)")
    print(f"Validation: {len(val_dataset)} ({len(val_dataset)/(len(train_dataset) + len(val_dataset))*100:.1f}%)")
    
    # Анализ длин токенов
    print(f"\nАнализ длин токенов (максимум 128):")
    
    # Считаем реальные длины (без padding)
    train_lengths = []
    for example in train_dataset.select(range(min(1000, len(train_dataset)))):  # Берем выборку для анализа
        # Считаем количество не-padding токенов
        real_length = sum(1 for token_id in example['input_ids'] if token_id != 50256)  # 50256 - pad_token_id для DialoGPT
        train_lengths.append(real_length)
    
    if train_lengths:
        avg_length = sum(train_lengths) / len(train_lengths)
        min_length = min(train_lengths)
        max_length = max(train_lengths)
        
        print(f"Средняя длина токенов: {avg_length:.1f}")
        print(f"Минимальная длина: {min_length}")
        print(f"Максимальная длина: {max_length}")
        
        # Распределение длин
        short_count = sum(1 for l in train_lengths if l < 64)
        long_count = sum(1 for l in train_lengths if l >= 64)
        
        print(f"\nРаспределение длин:")
        print(f"Короткие (< 64 токенов): {short_count} ({short_count/len(train_lengths)*100:.1f}%)")
        print(f"Средние (>= 64 токенов): {long_count} ({long_count/len(train_lengths)*100:.1f}%)")

def show_examples(train_dataset, num_examples=3):
    """
    Показывает примеры из датасета
    """
    print(f"\n=== Примеры из датасета (первые {num_examples}) ===")
    
    for i in range(min(num_examples, len(train_dataset))):
        example = train_dataset[i]
        print(f"\n--- Пример {i+1} ---")
        print(f"Prompt: {example['prompt']}")
        print(f"Chosen rating: {example['chosen_rating']}")
        print(f"Rejected rating: {example['rejected_rating']}")
        print(f"Длина токенов: {len([t for t in example['input_ids'] if t != 50256])}")

def load_processed_datasets(input_dir="processed_dataset"):
    """
    Загружает ранее сохраненные обработанные датасеты
    """
    from datasets import load_from_disk
    
    try:
        train_dataset = load_from_disk(f"{input_dir}/train")
        val_dataset = load_from_disk(f"{input_dir}/validation")
        
        print(f"✓ Загружены обработанные датасеты из {input_dir}/")
        print(f"Train: {len(train_dataset)} примеров")
        print(f"Validation: {len(val_dataset)} примеров")
        
        # Загружаем метаданные
        try:
            with open(f"{input_dir}/metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"✓ Метаданные загружены")
            print(f"Исходный датасет: {metadata['dataset_name']}")
            print(f"Максимальная длина токенов: {metadata['max_length']}")
            print(f"Токенизатор: {metadata['tokenizer']}")
        except:
            print("⚠ Метаданные не найдены")
            
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return None, None

def main():
    """
    Основная функция для демонстрации
    """
    print("=== Анализ обработанного датасета HelpSteer2_binarized ===\n")
    
    # Попробуем загрузить уже обработанные данные
    train_data, val_data = load_processed_datasets()
    
    if train_data is None:
        print("Обработанные данные не найдены. Запустите load_dataset.py сначала.")
        return
    
    # Анализируем статистику
    analyze_dataset_statistics(train_data, val_data)
    
if __name__ == "__main__":
    main()

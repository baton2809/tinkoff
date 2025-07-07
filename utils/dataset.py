import json
import os

from datasets import load_from_disk


def calculate_split_ratio(train_size, validate_size):
    total_size = train_size + validate_size
    train_percent = round((train_size / total_size) * 100)
    val_percent = round((validate_size / total_size) * 100)

    if train_percent + val_percent != 100:
        if train_percent > val_percent:
            train_percent = 100 - val_percent
        else:
            val_percent = 100 - train_percent

    return f"{train_percent}/{val_percent}"


def save_processed_dataset(train_dataset, val_dataset, max_length, batch_size,
                           dataset_name, tokenizer, output_dir="processed_dataset"):
    print(f"\nСохраняем обработанные датасеты в {output_dir}/...")

    os.makedirs(output_dir, exist_ok=True)

    train_dataset.save_to_disk(f"{output_dir}/train")
    val_dataset.save_to_disk(f"{output_dir}/validation")

    print(f"Train dataset сохранен в {output_dir}/train")
    print(f"Validation dataset сохранен в {output_dir}/validation")

    train_size = len(train_dataset)
    validation_size = len(val_dataset)

    metadata = {
        "dataset_name": dataset_name,
        "split_used": "average_rating",
        "train_size": train_size,
        "validation_size": validation_size,
        "max_length": max_length,
        "tokenizer": tokenizer,
        "columns": train_dataset.column_names,
        "split_ratio": calculate_split_ratio(train_size, validation_size),
        "batch_size": batch_size
    }

    with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Метаданные сохранены в {output_dir}/metadata.json")
    return output_dir


def load_processed_dataset(input_dir="processed_dataset"):
    try:
        train_dataset = load_from_disk(f"{input_dir}/train")
        val_dataset = load_from_disk(f"{input_dir}/validation")

        print(f"Загружены обработанные датасеты из {input_dir}/")
        print(f"Train: {len(train_dataset)} примеров")
        print(f"Validation: {len(val_dataset)} примеров")

        try:
            with open(f"{input_dir}/metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"Метаданные загружены")
            print(f"Исходный датасет: {metadata['dataset_name']}")
            print(f"Максимальная длина токенов: {metadata['max_length']}")
            print(f"Токенизатор: {metadata['tokenizer']}")
        except:
            print("Метаданные не найдены. Запусти `preprocess.load_dataset` для их создания.")

        return train_dataset, val_dataset, metadata

    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return None, None

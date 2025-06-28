# HelpSteer2 Dataset Processing

Этот проект содержит скрипты для загрузки, обработки и использования датасета HelpSteer2_binarized для обучения языковых моделей.

## 📋 Обзор

Проект успешно:
- ✅ Загрузил датасет `esfrankel17/original_HelpSteer2_binarized` с Hugging Face Hub
- ✅ Разделил данные на train (80%) и validation (20%) подвыборки
- ✅ Ограничил максимальную длину текста до 256 токенов для предотвращения OOM
- ✅ Токенизировал данные с помощью `microsoft/DialoGPT-medium`
- ✅ Сохранил обработанные датасеты для дальнейшего использования
- ✅ Оптимизировал для работы на M1 MacBook с MPS

## 📊 Статистика датасета

- **Общее количество примеров**: 8,678
- **Train dataset**: 6,942 примера (80%)
- **Validation dataset**: 1,736 примеров (20%)
- **Максимальная длина токенов**: 256
- **Средняя длина токенов**: 110.2
- **Средняя разница в рейтингах**: 0.76

### Распределение длин токенов:
- Короткие (< 64 токенов): 53.2%
- Средние (64-127 токенов): 9.1%
- Длинные (128-255 токенов): 9.2%
- Максимальные (256 токенов): 28.5%

## 🗂️ Структура файлов

```
├── load_helpsteer_dataset.py    # Основной скрипт загрузки и обработки
├── dataset_summary.py           # Анализ и сводка по датасету
├── usage_example.py            # Примеры использования
├── setup_libraries.py          # Настройка библиотек для M1 MacBook
├── requirements.txt            # Зависимости проекта
├── processed_helpsteer/        # Обработанные датасеты
│   ├── train/                  # Train dataset
│   ├── validation/             # Validation dataset
│   └── metadata.json           # Метаданные
└── README_helpsteer.md         # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Загрузка и обработка датасета
```bash
python load_helpsteer_dataset.py
```

### 3. Анализ обработанных данных
```bash
python dataset_summary.py
```

### 4. Примеры использования
```bash
python usage_example.py
```

## 💻 Использование в коде

### Загрузка обработанных данных
```python
from datasets import load_from_disk
import json

# Загружаем датасеты
train_dataset = load_from_disk("processed_helpsteer/train")
val_dataset = load_from_disk("processed_helpsteer/validation")

# Загружаем метаданные
with open("processed_helpsteer/metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Train: {len(train_dataset)} примеров")
print(f"Validation: {len(val_dataset)} примеров")
```

### Использование с DataLoader
```python
from torch.utils.data import DataLoader
import torch

def collate_fn(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
        'chosen_ratings': torch.tensor([item['chosen_rating'] for item in batch]),
        'rejected_ratings': torch.tensor([item['rejected_rating'] for item in batch])
    }

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
```

### Фильтрация по качеству предпочтений
```python
# Фильтруем примеры с высокой разницей в рейтингах
def filter_high_preference_gap(example):
    gap = example['chosen_rating'] - example['rejected_rating']
    return gap >= 0.5

filtered_train = train_dataset.filter(filter_high_preference_gap)
print(f"Отфильтровано: {len(filtered_train)} из {len(train_dataset)} примеров")
```

## 🎯 Применение для обучения

### Supervised Fine-Tuning (SFT)
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=True,
    dataloader_pin_memory=False,  # Для MPS
)
```

### Direct Preference Optimization (DPO)
```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo_results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    fp16=True,
)
```

## 🔧 Оптимизации для M1 MacBook

- **MPS поддержка**: Автоматическое использование Metal Performance Shaders
- **Экономия памяти**: fp16, маленькие batch sizes, gradient accumulation
- **Ограничение длины**: Максимум 256 токенов для предотвращения OOM
- **Эффективная токенизация**: Батчевая обработка с progress bars

## 📈 Структура данных

Каждый пример содержит:
- `prompt`: Исходный запрос пользователя
- `chosen`: Предпочтительный ответ (JSON формат)
- `chosen_rating`: Рейтинг предпочтительного ответа
- `rejected`: Отклоненный ответ (JSON формат)
- `rejected_rating`: Рейтинг отклоненного ответа
- `input_ids`: Токенизированный prompt (256 токенов)
- `attention_mask`: Маска внимания (256 токенов)

## 🎛️ Настройки

### Изменение максимальной длины токенов
```python
# В load_helpsteer_dataset.py
tokenized_train, tokenized_val = process_dataset(
    train_dataset, 
    val_dataset, 
    tokenizer, 
    max_length=512  # Измените на нужное значение
)
```

### Изменение соотношения train/validation
```python
# В load_helpsteer_dataset.py
train_dataset, val_dataset = split_dataset(
    dataset, 
    target_split, 
    test_size=0.1,  # 90/10 вместо 80/20
    seed=42
)
```

### Использование другого токенизатора
```python
# В load_helpsteer_dataset.py
tokenizer = setup_tokenizer("microsoft/DialoGPT-small")  # Или другая модель
```

## 🔍 Анализ качества данных

Датасет содержит примеры с различным качеством предпочтений:
- **Высокое качество** (разница ≥ 1.0): ~25% примеров
- **Среднее качество** (разница 0.5-1.0): ~35% примеров  
- **Низкое качество** (разница < 0.5): ~40% примеров

Рекомендуется фильтровать примеры с разницей в рейтингах ≥ 0.5 для лучшего качества обучения.

## 🚨 Требования к системе

- **Python**: 3.8+
- **PyTorch**: 2.0+ с MPS поддержкой
- **Transformers**: 4.30+
- **Datasets**: 2.10+
- **Память**: Минимум 8GB RAM (рекомендуется 16GB+)
- **Диск**: ~500MB для обработанных данных

## 📝 Примечания

1. **OOM предотвращение**: Ограничение до 256 токенов критично для M1 MacBook
2. **Batch size**: Рекомендуется использовать batch_size=1 с gradient_accumulation_steps=4
3. **Сохранение**: Обработанные данные сохраняются локально для быстрого доступа
4. **Фильтрация**: Доступна фильтрация по качеству предпочтений

## 🤝 Поддержка

Если возникают проблемы:
1. Проверьте доступность MPS: `torch.mps.is_available()`
2. Убедитесь в наличии свободной памяти
3. Попробуйте уменьшить max_length до 128 токенов
4. Используйте gradient_checkpointing для экономии памяти

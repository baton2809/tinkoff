# Настройка ML окружения для M1 MacBook

## Установка и настройка

### 1. Создание виртуального окружения
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 2. Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Проверка установки
```bash
python setup_libraries.py
```

## Особенности для M1 MacBook

### Использование MPS (Metal Performance Shaders)
- Автоматическое определение и использование MPS для ускорения на M1/M2 чипах
- Оптимизация памяти с использованием `torch.float16`
- Настройка batch size и gradient accumulation для эффективного использования памяти

### Оптимизации памяти
- `torch.float16` для экономии памяти
- Маленький batch size (1) с gradient accumulation (4)
- Отключение `dataloader_pin_memory` для совместимости с MPS
- Автоматическая очистка кэша MPS

## Основные библиотеки

- **torch**: Основной фреймворк для глубокого обучения
- **transformers**: Библиотека Hugging Face для работы с трансформерами
- **datasets**: Загрузка и обработка датасетов
- **peft**: Parameter-Efficient Fine-Tuning (LoRA, AdaLoRA и др.)
- **trl**: Transformer Reinforcement Learning (PPO, SFT)
- **accelerate**: Распределенное обучение и оптимизация

## Функции в setup_libraries.py

### `setup_model_and_tokenizer(model_name)`
Загружает модель и токенизатор с оптимизацией для M1:
```python
model, tokenizer = setup_model_and_tokenizer("microsoft/DialoGPT-medium")
```

### `setup_lora_config()`
Создает конфигурацию LoRA для эффективного fine-tuning:
```python
lora_config = setup_lora_config()
model = get_peft_model(model, lora_config)
```

### `setup_training_arguments()`
Настраивает аргументы обучения для M1:
```python
training_args = setup_training_arguments()
```

### `setup_accelerator()`
Настраивает Accelerator для оптимизации:
```python
accelerator = setup_accelerator()
```

## Пример использования

```python
from setup_libraries import *

# Проверка библиотек
check_libraries()

# Загрузка модели
model, tokenizer = setup_model_and_tokenizer("microsoft/DialoGPT-small")

# Настройка LoRA
lora_config = setup_lora_config()
model = get_peft_model(model, lora_config)

# Настройка обучения
training_args = setup_training_arguments()
accelerator = setup_accelerator()

print(f"Модель загружена на устройство: {device}")
print(f"Используется тип данных: {torch_dtype}")
```

## Рекомендации для M1 MacBook

1. **Память**: Используйте маленькие модели (до 7B параметров) для комфортной работы
2. **Batch Size**: Начинайте с batch_size=1 и увеличивайте gradient_accumulation_steps
3. **Модели**: Рекомендуемые модели для начала:
   - `microsoft/DialoGPT-small`
   - `distilgpt2`
   - `gpt2`
4. **Мониторинг**: Следите за использованием памяти через Activity Monitor
5. **Охлаждение**: При длительном обучении следите за температурой устройства

## Устранение проблем

### Ошибки памяти
- Уменьшите batch_size до 1
- Увеличьте gradient_accumulation_steps
- Используйте более маленькую модель
- Очистите кэш: `torch.backends.mps.empty_cache()`

### Проблемы с MPS
- Убедитесь, что используете macOS 12.3+
- Проверьте совместимость операций с MPS
- При необходимости переключитесь на CPU: `device = torch.device("cpu")`

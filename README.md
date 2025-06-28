# Тестовое задание в команду Alignment

## Описание

Reward Model - это модель, которая оценивает качество ответов на основе пар "хороший ответ" / "плохой ответ". Она используется в RLHF (Reinforcement Learning from Human Feedback) для обучения языковых моделей генерировать более качественные ответы.

## Начало

### 1. Создание окружения
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Проверить библиотеки
```bash
python setup_check.py
```

## Особенности для M1 MacBook

### Как проверить доступность MPS (Metal Performance Shaders)
```python
    import torch
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
```

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

## Статистика датасета

- **Общее количество примеров**: 8,678
- **Train dataset**: 6,942 примера (80%)
- **Validation dataset**: 1,736 примеров (20%)
- **Максимальная длина токенов**: 256
- **Средняя длина токенов**: 110.2
- **Средняя разница в рейтингах**: 0.76

### Распределение длин токенов
- Короткие (< 64 токенов): 53.2%
- Средние (64-127 токенов): 9.1%
- Длинные (> 127 токенов): 37.7%

## Выводы

- ✅ Загрузил датасет `
juyoungml/original_HelpSteer2_binarized` с Hugging Face Hub
- ✅ Разделил данные на train (80%) и validation (20%) подвыборки
- ✅ Ограничил максимальную длину текста до 256 токенов для предотвращения OOM
- ✅ Токенизировал данные с помощью `microsoft/DialoGPT-medium`
- ✅ Сохранил обработанные датасеты для дальнейшего использования
- ✅ Оптимизировал для работы на M1 MacBook с MPS

## Формат данных

Reward Model ожидает данные в формате:

```python
{
    "prompt": "Вопрос пользователя",
    "chosen": "Хороший ответ",
    "rejected": "Плохой ответ"
}
```

Во время обучения создаются пары:
- `prompt + chosen` → высокий reward score
- `prompt + rejected` → низкий reward score

# Reward Model Training

Этот проект содержит код для обучения Reward Model на основе SFT модели SmolLM2-135M-Instruct с использованием библиотеки TRL и датасета HelpSteer2_binarized.

## Описание

Reward Model - это модель, которая оценивает качество ответов на основе пар "хороший ответ" / "плохой ответ". Она используется в RLHF (Reinforcement Learning from Human Feedback) для обучения языковых моделей генерировать более качественные ответы.

## Структура проекта

```
├── train_reward_model.py      # Основной скрипт для обучения reward model
├── reward_model_demo.py       # Демонстрация использования обученной модели
├── load_smollm_model.py       # Загрузка базовой SFT модели
├── load_helpsteer_dataset.py  # Загрузка и обработка датасета
├── processed_helpsteer/       # Обработанный датасет
└── reward_model_output/       # Выходная директория для обученной модели
```

## Требования

Убедитесь, что установлены все необходимые библиотеки:

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `trl>=0.4.0`
- `datasets>=2.10.0`

## Пошаговое использование

### 1. Подготовка данных

Сначала загрузите и обработайте датасет:

```bash
python load_helpsteer_dataset.py
```

Этот скрипт:
- Загружает датасет HelpSteer2_binarized
- Разделяет на train/validation (80/20)
- Токенизирует тексты с максимальной длиной 256 токенов
- Сохраняет обработанные данные в `processed_helpsteer/`

### 2. Обучение Reward Model

Запустите обучение reward model:

```bash
python train_reward_model.py
```

Параметры обучения:
- **Базовая модель**: `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Архитектура**: `AutoModelForSequenceClassification` с `num_labels=1`
- **Learning rate**: `5e-5`
- **Epochs**: `1`
- **FP16**: `True`
- **Max length**: `256`
- **Batch size**: `4` (с gradient accumulation = 2)

Процесс обучения:
1. Загружает SFT модель как sequence classification model
2. Подготавливает датасет с парами chosen/rejected
3. Использует `RewardTrainer` из библиотеки TRL
4. Сохраняет обученную модель в `./reward_model_output/`

### 3. Тестирование модели

После обучения протестируйте модель:

```bash
python reward_model_demo.py
```

Демонстрация включает:
- Сравнение качества разных ответов на один вопрос
- Оценку диалогов
- Ранжирование ответов по качеству

## Архитектура Reward Model

```python
# Базовая модель
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    num_labels=1,  # Один выход для reward score
    torch_dtype=torch.float16
)

# Обучение с RewardTrainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args
)
```

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

## Использование обученной модели

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained("./reward_model_output")
tokenizer = AutoTokenizer.from_pretrained("./reward_model_output")

# Оценка ответа
prompt = "Как приготовить пасту?"
response = "Вскипятите воду, добавьте соль, варите пасту согласно инструкции."

text = f"{prompt}\n{response}"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    reward_score = model(**inputs).logits.item()

print(f"Reward score: {reward_score:.4f}")
```

## Класс RewardModelEvaluator

Для удобства использования создан класс `RewardModelEvaluator`:

```python
from reward_model_demo import RewardModelEvaluator

# Инициализация
evaluator = RewardModelEvaluator("./reward_model_output")

# Оценка одного ответа
score = evaluator.get_reward_score("Вопрос", "Ответ")

# Сравнение нескольких ответов
responses = ["Ответ 1", "Ответ 2", "Ответ 3"]
ranked = evaluator.compare_responses("Вопрос", responses)

# Оценка диалога
conversation = [("Вопрос 1", "Ответ 1"), ("Вопрос 2", "Ответ 2")]
scores = evaluator.evaluate_conversation(conversation)
```

## Мониторинг обучения

Во время обучения логируются:
- Loss на train и validation
- Шаги обучения каждые 50 итераций
- Сохранение чекпоинтов каждые 200 шагов
- Оценка на validation каждые 100 шагов

## Настройка параметров

Основные параметры можно изменить в `train_reward_model.py`:

```python
training_args = TrainingArguments(
    learning_rate=5e-5,           # Скорость обучения
    num_train_epochs=1,           # Количество эпох
    fp16=True,                    # Использование FP16
    per_device_train_batch_size=4, # Размер батча
    gradient_accumulation_steps=2, # Накопление градиентов
    max_grad_norm=1.0,            # Клиппинг градиентов
)
```

## Применение

Обученная Reward Model может использоваться для:

1. **RLHF обучения** - как reward function для PPO
2. **Фильтрации ответов** - отбор лучших ответов из нескольких вариантов
3. **Оценки качества** - автоматическая оценка качества генерации
4. **A/B тестирования** - сравнение разных версий моделей
5. **Ранжирования** - сортировка ответов по качеству

## Ограничения

- Модель обучена на конкретном датасете и может не обобщаться на другие домены
- Размер модели (135M параметров) ограничивает сложность оценки
- Требует качественных данных с парами chosen/rejected для хорошей производительности

## Дальнейшее развитие

Возможные улучшения:
- Использование более крупных базовых моделей
- Обучение на нескольких эпохах с learning rate scheduling
- Добавление регуляризации и dropout
- Использование ансамбля reward models
- Интеграция с PPO для полного RLHF пайплайна

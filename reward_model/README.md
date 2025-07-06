Для тестирования модели используй [README](../README.md) из корневой директории проекта.

Ниже описаны шаги для локальной отладки/тестирования

---

### Обучение модели

```shell
cd reward_model
python train.py
```

### Тестирование

#### Интерактивный режим

```shell
python inference.py --interactive
```

#### Оценка промпта

```shell
python inference.py --text "Your text here"
```

#### Сравнение ответов

```shell
python inference.py --prompt "How to cook pasta?" --response_a "Boil water, add salt, cook pasta according to instructions." --response_b "I don't know."
```

### Конфигурация хранится в `config/training_config.py`.

Подставь свои значения, если требуется настройка

### API

```python
from reward_model.src.model_manager import RewardModelManager

manager = RewardModelManager()
model, tokenizer = manager.load_trained_model("models/trained_model")

score = manager.get_reward_score("Boil water, add salt, cook pasta according to instructions")

result = manager.compare_responses(
    prompt="How to cook pasta?",
    response_a="Boil water, add salt, cook pasta according to instructions.",
    response_b="I don't know."
)
```
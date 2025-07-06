Для тестирования модели используй [README](../README.md) из корневой директории проекта.

Ниже описаны шаги для локальной отладки/тестирования

---

### Загружаем и токенизируем датасет `juyoungml/HelpSteer2-binarized`

```python -m preprocess.load_dataset```

```
DatasetDict({
    train: Dataset({
        features: ['prompt', 'chosen', 'rejected', 'chosen_score', 'rejected_score', 'chosen_rationale', 'rejected_rationale', 'score_diff', 'difficulty'],
        num_rows: 7224
    })
    validation: Dataset({
        features: ['prompt', 'chosen', 'rejected', 'chosen_score', 'rejected_score', 'chosen_rationale', 'rejected_rationale', 'score_diff', 'difficulty'],
        num_rows: 373
    })
})
```

### Запускаем обучение Reward Model

```python -m preprocess.train_reward_model```

Выбери `3` когда в консоли появится:

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice:
```
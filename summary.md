### Отчет

Разбил проект на 3 части:
- модуль `preprocess` - здесь логика загрузка и токенизации датасета, результаты сохраняются в `preprocessed_data`
- модуль `reward_model` - логика обучения Reward Model, на вход берет `preprocessed_data`, после обучения кладет модель в `trained_model`
- модуль `reinforce_baseline` - реализовано обучени, валидация и сравнение Reinforce baseline alignment с RW предыдущего шага.

Запускать скрипты в [README.md](README.md)

Пример лога обучения Reward Model:

```
Пример 373/373:
Промпт: ‘Plea negotiation is widely used in the criminal justice system, yet seldom praised.’ Evaluate the w...
Принятый ответ: Plea negotiation is a widely used procedure in the criminal justice system that allows defendants to...
Отвергнутый ответ: Plea bargaining, which is the process by which the defendant pleads guilty to a lesser charge or to ...
Оценка принятого ответа: 0.2835
Оценка отвергнутого ответа: -3.1477
Разница: 3.4312
Правильное предпочтение: Да
Точность: 30.29% (113/373)
Средняя разница в оценках: -0.0781
Количество оцененных примеров: 373

Обучение завершено

```

Пример лога обучения Reinforce:

```
Episode 5: Reward=-0.173, KL=6.672, Advantage=-0.238, Baseline=-0.767
Episode 10: Reward=-0.314, KL=7.029, Advantage=-0.192, Baseline=-0.858
Модель улучшена: Вознаграждение на валидации: -0.9645
Episode 15: Reward=0.617, KL=4.880, Advantage=0.117, Baseline=-0.084
Episode 20: Reward=-0.566, KL=5.848, Advantage=0.078, Baseline=-1.129
Модель улучшена: Вознаграждение на валидации: -0.6654
Episode 25: Reward=0.203, KL=7.123, Advantage=-0.025, Baseline=-0.449
Episode 30: Reward=-0.173, KL=3.933, Advantage=-0.287, Baseline=-0.459
Модель улучшена: Вознаграждение на валидации: -0.5891
Episode 35: Reward=-0.800, KL=5.086, Advantage=-0.669, Baseline=-0.679
Episode 40: Reward=0.486, KL=4.641, Advantage=0.338, Baseline=-0.233
Episode 45: Reward=0.476, KL=4.990, Advantage=0.250, Baseline=-0.152
Episode 50: Reward=0.809, KL=4.049, Advantage=0.597, Baseline=-0.165

Обучение завершено
Финальный baseline: -0.1648

Наилучшее вознаграждение на валидации: -0.5891 на эпизоде 30

Оценка после обучения

Результаты обучения:
Среднее вознаграждение до обучения: 0.2252
Среднее вознаграждение после обучения: -0.3488
Улучшение: -0.5740
Модель не показала улучшения

```

## Выводы
- Удалось реализовать Reward Model с качеством в ~36% (на 50 валидирующих примерах).
- Если RW модель уже обучена ранее, то подгружаем ее в код. Это поможет запускаться этапи и быстрее валидировать решение.
- Не удалось добиться улучшения модели Reinforce model, возможно из-за того, что KL нужно было взять меньше 0.1. Это приводило к высоким штрафам.
  - Можно попробовать KL=0.01, но тогда потребуется больше времени и итераций
  - Требуется пересмотреть гради
  - Решение Reinforce получилось наспех сделанным, стоит внимательно пройтись по алгоритмам обучения и исправить.

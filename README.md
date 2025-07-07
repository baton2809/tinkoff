## Quick Start

### Подготовим окружение

```shell
python3.13 -m venv .venv;source .venv/bin/activate;pip install --upgrade pip;pip install -r requirements.txt;
```

### Загружаем и токенизируем датасет

```shell
python -m preprocess.load_dataset --max_length 128 --batch_size 8
```

Аргумент `max_length` опциональный, задает максимальную длину токенов в датасете

### Запускаем обучение `Reward Model`

Отключим wandb

```shell
wandb disabled
```

```shell
python reward_model/train.py
```

Reward Model будет сохранена в `trained_model`.

Потворный запуск `train.py` загрузит модель из `trained_model`.

### Запускаем обучение и сравнение `Reinforce baseline alignment`

```shell
python reinforce_baseline/train.py
```

### Очистите временные файлы

```shell
chmod +x cleanup.sh 
./cleanup.sh
```

### Отключить виртуальное окружение
```shell
deactivate
```


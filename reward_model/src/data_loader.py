import os
from typing import Tuple, Optional

from datasets import Dataset, load_from_disk


class RewardDataLoader:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_datasets(self, split_ratio: int = 19, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Не удалось найти {self.data_dir}. "
                                    "Запусти скрипт preprocess.load_dataset")

        train_dataset = load_from_disk(os.path.join(self.data_dir, "train"))
        val_dataset = load_from_disk(os.path.join(self.data_dir, "validation"))

        required_columns = ["prompt", "chosen", "rejected"]
        missing_columns = [col for col in required_columns if col not in train_dataset.column_names]

        if missing_columns:
            raise ValueError(f"Не нашел колонку: {missing_columns}")

        if max_samples is not None:
            if len(train_dataset) > max_samples:
                train_dataset = train_dataset.select(range(max_samples))

            val_max = min(max_samples // split_ratio, len(val_dataset))
            if len(val_dataset) > val_max:
                val_dataset = val_dataset.select(range(val_max))

        return train_dataset, val_dataset

import argparse

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from utils.dataset import save_processed_dataset

CANDIDATE = "juyoungml/HelpSteer2-binarized"
LLM = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_LENGTH = 128
BATCH_SIZE = 8


def load_and_explore_dataset() -> tuple[Dataset, Dataset]:
    dataset = load_dataset(CANDIDATE)
    print(f"Подвыборки: {list(dataset.keys())}")

    train_split = dataset[list(dataset.keys())[0]]
    validation_split = dataset[list(dataset.keys())[1]]

    print(f"Количество примеров {list(dataset.keys())[0]}: {len(train_split)}")
    print(f"Колонки: {train_split.column_names}")

    print(f"\nПример данных:")
    first_example = train_split[0]
    for key, value in first_example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value.replace("\n", " ")[:100]}...")
        else:
            print(f"{key}: {value}")

    return train_split, validation_split


def setup_tokenizer() -> AutoTokenizer:
    print(f"\nТокенизатор: {LLM}")

    tokenizer = AutoTokenizer.from_pretrained(LLM)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Размер словаря {LLM}: {tokenizer.vocab_size}")
    return tokenizer


def tokenize_multiple_columns(
        examples: dict,
        tokenizer: AutoTokenizer,
        max_length: int,
        columns: list[str]
) -> dict:
    """
    Токенизируем несколько колонок для reward model
    """
    result = {}
    for column in columns:
        if column in examples:
            tokenized = tokenizer(examples[column],
                                  truncation=True,
                                  padding="max_length",
                                  max_length=max_length,
                                  return_tensors=None)

            for key, value in tokenized.items():
                result[f"{column}_{key}"] = value
    return result


def prepare_for_training(dataset: Dataset) -> Dataset:
    columns_to_keep = ["prompt", "chosen", "rejected"]
    tokenized_columns = [col for col in dataset.column_names if
                         any(col.startswith(prefix) for prefix in ['chosen_', 'rejected_', 'prompt_']) and col.endswith(
                             ('_input_ids', '_attention_mask'))]

    if tokenized_columns:
        columns_to_keep.extend(tokenized_columns)
        print(f"Токенизированные колонки: {tokenized_columns}")

    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

    return dataset.remove_columns(columns_to_remove)


def process_dataset(
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_length: int,
        batch_size: int
) -> tuple[Dataset, Dataset]:
    print(f"\nОбрабатываем датасеты с максимальной длиной {max_length} токенов...")

    columns = ['prompt', 'chosen', 'rejected']
    print(f"Токенизируем колонки: {columns}")

    tokenized_train = train_dataset.map(lambda x: tokenize_multiple_columns(x, tokenizer, max_length, columns),
                                        batched=True, batch_size=batch_size,
                                        desc=f"Токенизация train dataset батчами {batch_size}")

    tokenized_val = val_dataset.map(lambda x: tokenize_multiple_columns(x, tokenizer, max_length, columns),
                                    batched=True, batch_size=batch_size,
                                    desc=f"Токенизация validation dataset батчами {batch_size}")

    print(f"Train dataset: {len(tokenized_train)} примеров")
    print(f"Validation dataset: {len(tokenized_val)} примеров")

    print("\nСтруктура токенизированных данных:")
    for key in tokenized_train.column_names:
        if key.endswith('_input_ids'):
            print(f"  {key}: {len(tokenized_train[0][key])} токенов")

    cleaned_train_dataset = prepare_for_training(tokenized_train)
    cleaned_val_dataset = prepare_for_training(tokenized_val)

    print(f"\nКолонки, которые будут сохранены:")
    print(cleaned_train_dataset.column_names)

    return cleaned_train_dataset, cleaned_val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Dataset")
    parser.add_argument("--max_length", default="128", help="length of the tokenized sequences")
    parser.add_argument("--batch_size", default="8", help="batch size for training")
    args = parser.parse_args()

    max_length = int(args.max_length) or MAX_LENGTH
    batch_size = int(args.batch_size) or BATCH_SIZE

    print(f"Загрузка и обработка датасета {CANDIDATE}\n")

    train_dataset, val_dataset = load_and_explore_dataset()

    tokenizer = setup_tokenizer()

    tokenized_train, tokenized_val = process_dataset(train_dataset, val_dataset, tokenizer, max_length, batch_size)

    save_processed_dataset(tokenized_train, tokenized_val, max_length, batch_size, CANDIDATE, LLM)

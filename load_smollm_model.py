import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    torch_dtype=torch.float16
)

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Установка pad_token равным eos_token
tokenizer.pad_token = tokenizer.eos_token

print(f"Модель загружена: {model.config.name_or_path}")
print(f"Токенизатор загружен, pad_token: {tokenizer.pad_token}")

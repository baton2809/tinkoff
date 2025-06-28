"""
Установка и импорт необходимых библиотек для работы с ML на M1 MacBook
Использование torch.float16 для экономии памяти
"""

# Импорт основных библиотек
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from trl import (
    SFTTrainer,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead
)
from accelerate import Accelerator

# Настройка для M1 MacBook - использование MPS (Metal Performance Shaders)
if torch.mps.is_available():
    device = torch.device("mps")
    print("Используется MPS (Metal Performance Shaders) для M1 MacBook")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется CUDA")
else:
    device = torch.device("cpu")
    print("Используется CPU")

# Настройка типа данных для экономии памяти
torch_dtype = torch.float16
print(f"Используется тип данных: {torch_dtype}")

# Настройка для оптимизации памяти на M1
torch.mps.empty_cache() if torch.mps.is_available() else None

def setup_model_and_tokenizer(model_name: str):
    """
    Функция для загрузки модели и токенизатора с оптимизацией для M1 MacBook
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Добавляем pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type != "mps" else None,
        trust_remote_code=True
    )
    
    # Для MPS перемещаем модель вручную
    if device.type == "mps":
        model = model.to(device)
    
    return model, tokenizer

def setup_lora_config():
    """
    Настройка LoRA для эффективного fine-tuning
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    return lora_config

def setup_training_arguments():
    """
    Настройка аргументов для обучения с оптимизацией для M1
    """
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Маленький batch size для M1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Компенсируем маленький batch size
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        fp16=True,  # Используем fp16 для экономии памяти
        dataloader_pin_memory=False,  # Отключаем для MPS
        remove_unused_columns=False,
    )
    return training_args

def setup_accelerator():
    """
    Настройка Accelerator для распределенного обучения
    """
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=4
    )
    return accelerator

# Проверка доступности библиотек
def check_libraries():
    """
    Проверка успешного импорта всех библиотек
    """
    libraries = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "peft": __import__("peft").__version__,
        "trl": __import__("trl").__version__,
        "accelerate": __import__("accelerate").__version__
    }
    
    print("Успешно импортированы библиотеки:")
    for lib, version in libraries.items():
        print(f"  {lib}: {version}")
    
    print(f"\nИспользуемое устройство: {device}")
    print(f"Тип данных: {torch_dtype}")
    
    return libraries

if __name__ == "__main__":
    print("Настройка библиотек для ML на M1 MacBook...")
    check_libraries()
    print("\nВсе библиотеки готовы к использованию!")

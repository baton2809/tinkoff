import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    print("\nУспешно установлены библиотеки:")
    for lib, version in libraries.items():
        print(f"  {lib}: {version}")

    # Настройка для M1 MacBook - использование MPS (Metal Performance Shaders)
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Настройка типа данных для экономии памяти
    torch_dtype = torch.float16

    print(f"\nИспользуемое устройство: {device}")
    print(f"Тип данных: {torch_dtype}")


if __name__ == "__main__":
    check_libraries()
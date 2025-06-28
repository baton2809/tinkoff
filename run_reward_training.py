"""
Быстрый запуск полного процесса обучения Reward Model
Автоматически выполняет все этапы: подготовка данных → обучение → тестирование
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """
    Выполняет команду и показывает прогресс
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Команда: {command}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✅ {description} завершено за {elapsed:.1f} секунд")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Ошибка в {description} после {elapsed:.1f} секунд")
        print(f"Код ошибки: {e.returncode}")
        return False

def check_requirements():
    """
    Проверяет наличие необходимых файлов и библиотек
    """
    print("🔍 Проверка требований...")
    
    # Проверяем наличие файлов
    required_files = [
        "requirements.txt",
        "load_dataset.py", 
        "train_reward_model.py",
        "test_reward_model.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        return False
    
    # Проверяем Python версию
    if sys.version_info < (3, 8):
        print(f"❌ Требуется Python 3.8+, текущая версия: {sys.version}")
        return False
    
    print("✅ Все требования выполнены")
    return True

def install_requirements():
    """
    Устанавливает необходимые библиотеки
    """
    if not os.path.exists("requirements.txt"):
        print("❌ Файл requirements.txt не найден")
        return False
    
    return run_command(
        "pip install -r requirements.txt",
        "Установка зависимостей"
    )

def prepare_dataset():
    """
    Подготавливает датасет для обучения
    """
    # Проверяем, есть ли уже обработанный датасет
    if os.path.exists("processed_dataset") and os.path.exists("processed_dataset/metadata.json"):
        print("✅ Обработанный датасет уже существует, пропускаем подготовку")
        return True
    
    return run_command(
        "python load_dataset.py",
        "Подготовка датасета HelpSteer2_binarized"
    )

def train_reward_model():
    """
    Обучает reward model
    """
    # Проверяем, есть ли уже обученная модель
    if os.path.exists("reward_model_output") and os.path.exists("reward_model_output/config.json"):
        print("⚠️  Обученная модель уже существует в reward_model_output/")
        response = input("Хотите переобучить модель? (y/N): ").strip().lower()
        if response not in ['y', 'yes', 'да']:
            print("✅ Пропускаем обучение, используем существующую модель")
            return True
    
    return run_command(
        "python train_reward_model.py",
        "Обучение Reward Model"
    )

def test_reward_model():
    """
    Тестирует обученную reward model
    """
    return run_command(
        "python test_reward_model.py",
        "Тестирование обученной Reward Model"
    )

def show_summary():
    """
    Показывает итоговую информацию
    """
    print(f"\n{'='*60}")
    print("🎉 ОБУЧЕНИЕ REWARD MODEL ЗАВЕРШЕНО!")
    print(f"{'='*60}")
    
    # Проверяем результаты
    if os.path.exists("reward_model_output"):
        model_files = list(Path("reward_model_output").glob("*"))
        print(f"📁 Модель сохранена в: reward_model_output/")
        print(f"📊 Файлов в модели: {len(model_files)}")
    
    if os.path.exists("processed_dataset"):
        print(f"📁 Датасет обработан в: processed_dataset/")
    
    print(f"\n📖 Документация: README_reward_model.md")
    print(f"🧪 Для тестирования: python test_reward_model.py")
    print(f"🔧 Для настройки: отредактируйте train_reward_model.py")
    
    print(f"\n💡 Следующие шаги:")
    print(f"   1. Изучите результаты тестирования выше")
    print(f"   2. Используйте RewardModelEvaluator для своих задач")
    print(f"   3. Интегрируйте reward model в RLHF пайплайн")

def main():
    """
    Основная функция для запуска полного процесса
    """
    print("🤖 АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ REWARD MODEL")
    print("Базовая модель: SmolLM2-135M-Instruct")
    print("Датасет: HelpSteer2_binarized")
    print("Trainer: RewardTrainer (TRL)")
    
    # 1. Проверка требований
    if not check_requirements():
        print("\n❌ Проверка требований не пройдена")
        return False
    
    # 2. Установка зависимостей (опционально)
    install_deps = input("\n🔧 Установить/обновить зависимости? (y/N): ").strip().lower()
    if install_deps in ['y', 'yes', 'да']:
        if not install_requirements():
            print("\n❌ Ошибка при установке зависимостей")
            return False
    
    # 3. Подготовка датасета
    if not prepare_dataset():
        print("\n❌ Ошибка при подготовке датасета")
        return False
    
    # 4. Обучение модели
    if not train_reward_model():
        print("\n❌ Ошибка при обучении модели")
        return False
    
    # 5. Тестирование модели
    if not test_reward_model():
        print("\n❌ Ошибка при тестировании модели")
        return False
    
    # 6. Показываем итоги
    show_summary()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n🎯 Процесс завершен успешно!")
            exit(0)
        else:
            print(f"\n💥 Процесс завершен с ошибками")
            exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Процесс прерван пользователем")
        exit(1)
    except Exception as e:
        print(f"\n💥 Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

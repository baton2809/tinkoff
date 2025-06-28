"""
Быстрые решения для ошибки MPS memory
Запустите этот скрипт для немедленного применения исправлений
"""

import os
import sys

def apply_immediate_fixes():
    """Применяет немедленные исправления для MPS memory error"""
    print("🚀 Применение быстрых исправлений для MPS memory error\n")
    
    # 1. Установка переменных окружения
    print("1. Установка переменных окружения...")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    print("✓ Переменные окружения установлены")
    
    # 2. Очистка памяти
    print("\n2. Очистка памяти...")
    try:
        import torch
        import gc
        
        if torch.mps.is_available():
            torch.mps.empty_cache()
            print("✓ MPS кэш очищен")
        
        gc.collect()
        print("✓ Сборщик мусора запущен")
        
    except ImportError:
        print("⚠ PyTorch не найден, пропускаем очистку MPS")
    
    # 3. Создание команд для терминала
    print("\n3. Команды для терминала:")
    print("Выполните эти команды в терминале перед запуском обучения:")
    print("export PYTORCH_ENABLE_MPS_FALLBACK=1")
    print("export OMP_NUM_THREADS=1")
    print("export MKL_NUM_THREADS=1")
    
    # 4. Рекомендуемые параметры
    print("\n4. Рекомендуемые параметры для обучения:")
    print("- max_samples=200-500 (вместо полного датасета)")
    print("- batch_size=1 (вместо 2)")
    print("- max_length=64-128 (вместо 256)")
    print("- gradient_accumulation_steps=8-16")
    
    print("\n✅ Быстрые исправления применены!")
    print("Теперь запустите: python train_reward_model_optimized.py")

if __name__ == "__main__":
    apply_immediate_fixes()

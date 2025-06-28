"""
Скрипт для диагностики и решения проблем с памятью MPS на Apple Silicon
"""

import os
import torch
import psutil
import gc

def check_system_memory():
    """Проверяет состояние системной памяти"""
    print("=== Состояние системной памяти ===")
    
    # Системная память
    memory = psutil.virtual_memory()
    print(f"Общая память: {memory.total / (1024**3):.1f} GB")
    print(f"Доступная память: {memory.available / (1024**3):.1f} GB")
    print(f"Используется: {memory.percent}%")
    print(f"Свободно: {memory.free / (1024**3):.1f} GB")
    
    # MPS память (если доступна)
    if torch.mps.is_available():
        print(f"\n=== MPS устройство ===")
        print(f"MPS доступен: {torch.mps.is_available()}")
        try:
            # Попытка получить информацию о памяти MPS
            torch.mps.empty_cache()
            print("✓ MPS кэш очищен")
        except Exception as e:
            print(f"⚠ Ошибка при работе с MPS: {e}")

def apply_memory_fixes():
    """Применяет исправления для экономии памяти"""
    print("\n=== Применение исправлений памяти ===")
    
    # 1. Переменные окружения для PyTorch MPS
    fixes_applied = []
    
    # Включить fallback на CPU при ошибках MPS
    if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        fixes_applied.append("PYTORCH_ENABLE_MPS_FALLBACK=1")
    
    # Отключить верхний лимит памяти MPS (осторожно!)
    # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    # fixes_applied.append("PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
    
    # Ограничить количество потоков
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'
        fixes_applied.append("OMP_NUM_THREADS=1")
    
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '1'
        fixes_applied.append("MKL_NUM_THREADS=1")
    
    # 2. Очистка памяти
    gc.collect()
    if torch.mps.is_available():
        torch.mps.empty_cache()
    
    print("✓ Применены исправления:")
    for fix in fixes_applied:
        print(f"  - {fix}")
    
    return fixes_applied

def test_memory_allocation():
    """Тестирует выделение памяти на MPS"""
    print("\n=== Тест выделения памяти ===")
    
    if not torch.mps.is_available():
        print("❌ MPS недоступен")
        return False
    
    try:
        # Тест с небольшим тензором
        device = torch.device("mps")
        test_tensor = torch.randn(100, 100, device=device)
        print("✓ Небольшой тензор (100x100) создан успешно")
        
        # Тест с более крупным тензором
        test_tensor2 = torch.randn(1000, 1000, device=device)
        print("✓ Средний тензор (1000x1000) создан успешно")
        
        # Очистка
        del test_tensor, test_tensor2
        torch.mps.empty_cache()
        print("✓ Память очищена")
        
        return True
        
    except RuntimeError as e:
        if "MPS backend out of memory" in str(e):
            print("❌ Ошибка нехватки памяти MPS при тестировании")
            return False
        else:
            print(f"❌ Другая ошибка MPS: {e}")
            return False

def recommend_solutions():
    """Рекомендует решения проблем с памятью"""
    print("\n=== Рекомендации по решению проблем ===")
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    print("1. НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:")
    print("   - Закройте браузер и другие тяжелые приложения")
    print("   - Перезапустите Python/Jupyter для очистки памяти")
    print("   - Используйте оптимизированный скрипт: train_reward_model_optimized.py")
    
    print("\n2. ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print("   - Уменьшите max_samples до 100-500")
    print("   - Установите batch_size=1")
    print("   - Уменьшите max_length до 64-128")
    print("   - Отключите gradient_checkpointing")
    
    print("\n3. ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:")
    print("   export PYTORCH_ENABLE_MPS_FALLBACK=1")
    print("   export OMP_NUM_THREADS=1")
    print("   export MKL_NUM_THREADS=1")
    
    if available_gb < 8:
        print("\n⚠ КРИТИЧЕСКИ МАЛО ПАМЯТИ!")
        print("4. АЛЬТЕРНАТИВЫ:")
        print("   - Используйте CPU вместо MPS")
        print("   - Обучайте модель частями")
        print("   - Используйте облачные сервисы (Colab, Kaggle)")
    
    print("\n5. ЭКСТРЕННОЕ РЕШЕНИЕ:")
    print("   Добавьте в начало скрипта:")
    print("   os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'")
    print("   ⚠ ВНИМАНИЕ: Может привести к зависанию системы!")

def create_cpu_fallback_script():
    """Создает скрипт для обучения на CPU"""
    cpu_script = """
# Добавьте в начало вашего скрипта для принудительного использования CPU
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# или полностью отключить MPS:
# os.environ['PYTORCH_MPS_DISABLE'] = '1'

import torch
# Принудительно использовать CPU
torch.set_default_device('cpu')
"""
    
    with open('cpu_fallback_setup.py', 'w', encoding='utf-8') as f:
        f.write(cpu_script)
    
    print("\n✓ Создан файл cpu_fallback_setup.py")
    print("  Импортируйте его в начале вашего скрипта для использования CPU")

def main():
    """Основная функция диагностики"""
    print("🔧 Диагностика и исправление проблем с памятью MPS\n")
    
    # 1. Проверяем состояние памяти
    check_system_memory()
    
    # 2. Применяем исправления
    apply_memory_fixes()
    
    # 3. Тестируем выделение памяти
    mps_works = test_memory_allocation()
    
    # 4. Даем рекомендации
    recommend_solutions()
    
    # 5. Создаем fallback скрипт
    create_cpu_fallback_script()
    
    print("\n=== ИТОГОВЫЕ РЕКОМЕНДАЦИИ ===")
    if mps_works:
        print("✅ MPS работает, используйте train_reward_model_optimized.py")
        print("   с параметрами max_samples=500, batch_size=1")
    else:
        print("❌ MPS не работает, рекомендуется:")
        print("   1. Использовать CPU (импортируйте cpu_fallback_setup.py)")
        print("   2. Или применить PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        print("   3. Или использовать облачные сервисы")

if __name__ == "__main__":
    main()

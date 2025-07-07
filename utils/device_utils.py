import torch
import gc


def clear_memory():
    if torch.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def clear_memory_before(func):
    def wrapper(*args, **kwargs):
        clear_memory()
        return func(*args, **kwargs)
    return wrapper


def get_device(verbose=True):
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if verbose:
        print(f"Используется устройство: {device}")
    
    return device


def get_device_info():
    info = {
        'mps_available': torch.mps.is_available() if hasattr(torch, 'mps') else False,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': get_device(verbose=False)
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    return info


def get_optimal_dtype_and_device_map():
    if torch.mps.is_available():
        torch_dtype = torch.float32
        device_map = None
        return torch_dtype, device_map
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
        return torch_dtype, device_map
    else:
        torch_dtype = torch.float32
        device_map = None
        return torch_dtype, device_map


def get_model_loading_config(verbose=True):
    torch_dtype, device_map = get_optimal_dtype_and_device_map()
    
    config = {
        'torch_dtype': torch_dtype,
        'device_map': device_map,
        'low_cpu_mem_usage': True,
    }
    
    if verbose:
        if torch.mps.is_available():
            print("Используется torch.float32 для MPS устройства")
            print("Device_map отключен для лучшей совместимости с MPS")
        elif torch.cuda.is_available():
            print("Используется torch.float16 для CUDA устройства")
            print("Device_map установлен в 'auto'")
        else:
            print("Используется torch.float32 для CPU устройства")
    
    return config





# Добавьте в начало вашего скрипта для принудительного использования CPU
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# или полностью отключить MPS:
# os.environ['PYTORCH_MPS_DISABLE'] = '1'

import torch
# Принудительно использовать CPU
torch.set_default_device('cpu')

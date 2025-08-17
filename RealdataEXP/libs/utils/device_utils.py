"""
设备选择和管理工具
"""
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# --- 关键修复：提前在顶层导入IPEX ---
# 我们在程序早期就尝试导入IPEX，避免与其他库（如numpy, pandas）产生底层DLL冲突。
_IPEX_AVAILABLE = False
try:
    import intel_extension_for_pytorch as ipex
    _IPEX_AVAILABLE = True
    # 这条日志现在应该会在程序启动时很早就出现
    logger.info("Intel Extension for PyTorch (IPEX) discovered and imported successfully at top level.")
except ImportError:
    # 如果这里失败，说明环境确实有问题
    logger.info("Intel Extension for PyTorch (IPEX) not found during initial import. IPEX backend will be unavailable.")
# --- 修复结束 ---


def get_device_and_amp_helpers(device_choice='auto'):
    """
    Dynamically determines the best available device and corresponding AMP tools.
    """
    class StubScaler:
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def get_scale(self): return 1.0
        def is_enabled(self): return False

    @contextmanager
    def stub_autocast(device_type, *args, **kwargs):
        yield

    # 'auto' detection order: cuda -> ipex -> xpu -> dml -> cpu

    # 1. Check for CUDA
    if device_choice.lower() in ['auto', 'cuda']:
        try:
            if torch.cuda.is_available():
                from torch.amp import autocast, GradScaler
                device = torch.device("cuda")
                logger.info("[Device] CUDA is available. Using CUDA backend (Full AMP).")
                return device, autocast, GradScaler
        except ImportError:
            logger.warning("[Device] torch.cuda or torch.amp not found, skipping CUDA check.")

    # 2. Check for IPEX (Full Optimization)
    if device_choice.lower() in ['auto', 'ipex']:
        # 导入已在顶层完成，这里只检查标志和设备可用性
        if _IPEX_AVAILABLE and torch.xpu.is_available():
            # IPEX has autocast but not GradScaler. We return our StubScaler.
            from torch.xpu.amp import autocast
            device = torch.device("xpu")
            logger.info("[Device] Intel IPEX is available. Using XPU backend (Full IPEX Optimization & AMP).")
            # Return the REAL autocast but a FAKE scaler class
            return device, autocast, StubScaler
        elif device_choice.lower() == 'ipex':
            # 处理用户明确要求'ipex'但顶层导入失败的情况
            logger.warning("[Device] 'ipex' was chosen, but the IPEX library could not be imported or is not functional.")

    # 3. Check for XPU (Basic Device Placement)
    if device_choice.lower() in ['auto', 'xpu']:
        try:
            # 即使没有顶层导入成功，基础的xpu设备也可能被torch识别
            if torch.xpu.is_available():
                device = torch.device("xpu")
                logger.info("[Device] Intel XPU device is available. Using XPU backend (Basic, NO IPEX Optimizations, NO AMP).")
                return device, stub_autocast, StubScaler
        except (ImportError, AttributeError):
            if device_choice.lower() == 'xpu':
                 logger.warning("[Device] 'xpu' was chosen, but torch.xpu was not available.")

    # 4. Check for DirectML
    if device_choice.lower() in ['auto', 'dml']:
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                logger.info("[Device] DirectML is available. Using DML backend (NO AMP).")
                return device, stub_autocast, StubScaler
        except ImportError:
            if device_choice.lower() == 'dml':
                logger.warning("[Device] 'dml' was chosen, but torch_directml not found.")
            
    # 5. Fallback to CPU
    logger.info("[Device] No specified or available GPU backend found. Falling back to CPU.")
    device = torch.device("cpu")
    return device, stub_autocast, StubScaler
"""
设备选择和管理工具
"""
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_device_and_amp_helpers(device_choice='auto'):
    """
    Dynamically determines the best available device and corresponding AMP tools.
    Separates 'ipex' (full optimization) from 'xpu' (basic device placement).

    Args:
        device_choice (str): 'auto', 'cuda', 'ipex', 'xpu', 'dml', 'cpu'.

    Returns:
        tuple: (torch.device, autocast_context_manager, GradScalerClass)
    """

    # --- IMPORTANT: Helper definitions must be at the top of the function ---
    class StubScaler:
        """A virtual GradScaler that does nothing."""
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def get_scale(self): return 1.0
        def is_enabled(self): return False

    @contextmanager
    def stub_autocast(device_type, *args, **kwargs):
        """A virtual autocast context that does nothing."""
        yield
    # --- End of helper definitions ---

    # --- Detection Logic ---
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
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                from torch.xpu.amp import autocast, GradScaler
                device = torch.device("xpu")
                logger.info("[Device] Intel IPEX is available. Using XPU backend (Full IPEX Optimization & AMP).")
                return device, autocast, GradScaler
        except ImportError:
            if device_choice.lower() == 'ipex':
                logger.warning("[Device] 'ipex' was chosen, but Intel Extension for PyTorch not found.")

    # 3. Check for XPU (Basic Device Placement)
    if device_choice.lower() in ['auto', 'xpu']:
        try:
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
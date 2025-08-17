import sys
import os

# 模拟 main.py 的项目路径设置
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

print("--- Starting IPEX Import Test ---")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# --- 步骤 1: 检查 Torch ---
try:
    print("\nAttempting to import torch...")
    import torch
    print(f"Torch version: {torch.__version__}")
    print("SUCCESS: Torch import successful.")
except Exception as e:
    print(f"FATAL ERROR importing torch: {e}")
    sys.exit(1)

# --- 步骤 2: 检查 IPEX ---
try:
    print("\nAttempting to import intel_extension_for_pytorch...")
    import intel_extension_for_pytorch as ipex
    print(f"IPEX version: {ipex.__version__}")
    print("SUCCESS: IPEX import successful.")
except Exception as e:
    print(f"FATAL ERROR importing intel_extension_for_pytorch: {e}")
    # 打印更详细的路径信息，帮助诊断
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 步骤 3: 检查 XPU 可用性 ---
print("\nChecking torch.xpu.is_available()...")
try:
    available = torch.xpu.is_available()
    print(f"torch.xpu.is_available() returned: {available}")
    if not available:
        print("WARNING: IPEX imported but XPU device is not available!")
except Exception as e:
    print(f"ERROR checking torch.xpu.is_available(): {e}")

print("\n--- Test Finished ---")
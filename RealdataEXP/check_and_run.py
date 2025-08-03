#!/usr/bin/env python3
"""
环境检测并运行实验的Python脚本
提供更好的错误处理和用户交互
"""

import os
import sys
import subprocess
import torch
import time

def print_banner(title):
    """打印标题横幅"""
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_gpu_availability():
    """检查GPU可用性"""
    print("🔍 检查GPU设备...")
    
    # 检查nvidia-smi命令
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi 可用")
            # 显示GPU信息的简化版本
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'RTX' in line or 'GTX' in line or 'A30' in line or 'V100' in line:
                    print(f"   GPU: {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi 命令失败")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi 不可用")
        return False

def check_cuda_toolkit():
    """检查CUDA工具包"""
    print("\n🔍 检查CUDA工具包...")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 提取CUDA版本信息
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"✅ CUDA编译器: {line.strip()}")
            return True
        else:
            print("⚠️ CUDA编译器不可用")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ nvcc 命令不可用")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print("\n🔍 检查PyTorch CUDA支持...")
    
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("✅ PyTorch检测到CUDA支持")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
        
        # 简单的GPU计算测试
        try:
            device = torch.device('cuda')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print("✅ GPU计算测试通过")
            
            # 清理
            del x, y, z
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"❌ GPU计算测试失败: {e}")
            return False
    else:
        print("❌ PyTorch检测不到CUDA支持")
        return False

def check_project_setup():
    """检查项目配置"""
    print("\n🔍 检查项目配置...")
    
    # 检查配置文件
    config_file = "configs/experiment.yaml"
    if os.path.exists(config_file):
        print(f"✅ 配置文件存在: {config_file}")
    else:
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 检查主程序
    main_file = "main.py"
    if os.path.exists(main_file):
        print(f"✅ 主程序存在: {main_file}")
    else:
        print(f"❌ 主程序不存在: {main_file}")
        return False
    
    # 检查数据目录
    data_dir = "data/KuaiRand/Pure"
    if os.path.exists(data_dir):
        print(f"✅ 数据目录存在: {data_dir}")
        files = os.listdir(data_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"   找到 {len(csv_files)} 个CSV数据文件")
    else:
        print(f"⚠️ 数据目录不存在: {data_dir}")
        print("   注意：如果是首次运行，请确保数据文件存在")
    
    return True

def wait_for_user_confirmation():
    """等待用户确认"""
    print("\n" + "=" * 60)
    print("🚀 准备开始实验")
    print("=" * 60)
    print("所有环境检查已完成！")
    print()
    print("实验将使用以下配置:")
    print("  - 配置文件: configs/experiment.yaml")
    print("  - 实验模式: global")
    print("  - 设备: 自动选择 (GPU优先)")
    print()
    
    try:
        input("按回车键开始实验，或按Ctrl+C退出...")
    except KeyboardInterrupt:
        print("\n用户取消实验")
        sys.exit(0)

def run_experiment():
    """运行实验"""
    print("\n" + "=" * 60)
    print("🔥 开始运行实验")
    print("=" * 60)
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':' + os.getcwd()
    
    try:
        # 运行主程序
        result = subprocess.run([
            sys.executable, 'main.py', 
            '--config', 'configs/experiment.yaml', 
            '--mode', 'global'
        ], check=True)
        
        print("\n" + "=" * 60)
        print("🎉 实验成功完成！")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验运行失败，退出码: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户中断实验")
        sys.exit(0)

def main():
    """主函数"""
    print_banner("RealdataEXP GPU环境检测与实验启动")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"主机: {os.uname().nodename}")
    print(f"工作目录: {os.getcwd()}")
    
    # 环境检测步骤
    checks = [
        ("GPU设备", check_gpu_availability),
        ("CUDA工具包", check_cuda_toolkit), 
        ("PyTorch CUDA", check_pytorch_cuda),
        ("项目设置", check_project_setup)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print_banner(f"检查: {check_name}")
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            print(f"❌ 检查 {check_name} 时出错: {e}")
            failed_checks.append(check_name)
    
    # 总结检测结果
    print_banner("检测结果总结")
    if failed_checks:
        print("❌ 以下检查失败:")
        for check in failed_checks:
            print(f"   - {check}")
        
        # 只要PyTorch CUDA可用就可以继续
        if "PyTorch CUDA" not in failed_checks:
            print("\n⚠️ 尽管某些检查失败，但PyTorch CUDA可用")
            print("实验仍可能正常运行")
        else:
            print("\n🛑 关键检查失败，建议修复后再运行实验")
            choice = input("是否仍要继续? (y/N): ").strip().lower()
            if choice != 'y':
                print("退出程序")
                sys.exit(1)
    else:
        print("✅ 所有检查通过！")
    
    # 等待用户确认并运行实验
    wait_for_user_confirmation()
    run_experiment()

if __name__ == "__main__":
    main()
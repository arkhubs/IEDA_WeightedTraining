"""
GPU诊断和监控工具
用于检测GPU状态、内存使用和训练过程中的GPU利用率
"""

import logging
import torch
import time
import threading
import subprocess
import os

logger = logging.getLogger(__name__)

def log_gpu_info():
    """记录详细的GPU环境信息"""
    if not torch.cuda.is_available():
        logger.warning("[GPU检查] CUDA不可用，将使用CPU运行")
        return

    logger.info("========== GPU诊断信息 ==========")
    try:
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        logger.info(f"[GPU检查] CUDA可用，使用GPU: {device_name}")
        logger.info(f"[GPU检查]   - 设备ID: {device_id}")
        
        # 获取GPU属性
        props = torch.cuda.get_device_properties(device_id)
        total_mem = props.total_memory / (1024**3)
        logger.info(f"[GPU检查]   - 总内存: {total_mem:.2f} GB")
        logger.info(f"[GPU检查]   - 计算能力: {props.major}.{props.minor}")
        logger.info(f"[GPU检查]   - 多处理器数量: {props.multi_processor_count}")
        
        # 初始内存使用情况
        allocated_mem = torch.cuda.memory_allocated(device_id) / (1024**2)
        reserved_mem = torch.cuda.memory_reserved(device_id) / (1024**2)
        logger.info(f"[GPU检查]   - 初始已分配内存: {allocated_mem:.2f} MB")
        logger.info(f"[GPU检查]   - 初始保留内存: {reserved_mem:.2f} MB")
        
        # 测试GPU操作
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor)
        logger.info(f"[GPU检查]   - GPU运算测试: 通过 (1000x1000矩阵乘法)")
        
        # 清理测试张量
        del test_tensor, result
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"[GPU检查] 获取GPU详情失败: {e}")
    logger.info("=====================================")

def log_gpu_memory_usage(prefix=""):
    """记录当前GPU内存使用情况"""
    if not torch.cuda.is_available():
        return
    
    try:
        device_id = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**2)
        logger.info(f"[GPU内存{prefix}] 已分配: {allocated:.2f} MB, 保留: {reserved:.2f} MB")
    except Exception as e:
        logger.error(f"[GPU内存{prefix}] 获取内存信息失败: {e}")

def test_gpu_training_speed():
    """测试GPU训练速度"""
    if not torch.cuda.is_available():
        logger.warning("[GPU速度测试] CUDA不可用，跳过测试")
        return
    
    logger.info("[GPU速度测试] 开始GPU训练速度测试...")
    try:
        # 创建测试数据
        batch_size = 1024
        input_dim = 100
        hidden_dim = 256
        
        # 创建模型和数据
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        ).cuda()
        
        data = torch.randn(batch_size, input_dim).cuda()
        target = torch.randn(batch_size, 1).cuda()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 预热
        for _ in range(10):
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 正式测试
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(100):
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        samples_per_sec = (100 * batch_size) / elapsed
        
        logger.info(f"[GPU速度测试] 完成100次迭代，用时: {elapsed:.2f}秒")
        logger.info(f"[GPU速度测试] 处理速度: {samples_per_sec:.0f} 样本/秒")
        
        # 清理
        del model, data, target
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"[GPU速度测试] 测试失败: {e}")

class GPUMonitor:
    """GPU实时监控器"""
    
    def __init__(self, log_interval=30):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控GPU状态"""
        if not torch.cuda.is_available():
            logger.warning("[GPU监控] CUDA不可用，跳过监控")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"[GPU监控] 开始监控，每{self.log_interval}秒记录一次")
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("[GPU监控] 停止监控")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                log_gpu_memory_usage(" - 监控")
                
                # 尝试获取GPU利用率
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        gpu_util = float(result.stdout.strip())
                        logger.info(f"[GPU监控] GPU利用率: {gpu_util}%")
                except Exception:
                    pass  # 如果nvidia-smi不可用，跳过利用率检查
                    
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"[GPU监控] 监控过程出错: {e}")
                break

def setup_gpu_monitoring(log_interval=30):
    """设置GPU监控"""
    monitor = GPUMonitor(log_interval)
    monitor.start_monitoring()
    return monitor

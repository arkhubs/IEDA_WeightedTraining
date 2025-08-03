class Logger:
    """
    日志记录器类，封装了日志记录功能
    """
    
    def __init__(self, log_file=None, log_level=logging.INFO):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径，如果为None则只输出到控制台
            log_level: 日志级别
        """
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # 清除现有处理器
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg):
        """记录信息级别的日志"""
        self.logger.info(msg)
    
    def warning(self, msg):
        """记录警告级别的日志"""
        self.logger.warning(msg)
    
    def error(self, msg):
        """记录错误级别的日志"""
        self.logger.error(msg)
    
    def debug(self, msg):
        """记录调试级别的日志"""
        self.logger.debug(msg)


def setup_seed(seed: int) -> None:
    """
    设置随机种子，确保实验可重复
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

"""
模型模块
包含多标签预测模型和损失函数
"""

from .mlp_model import MLPModel
from .multi_label_model import MultiLabelModel
from .loss_functions import LogMAELoss, get_loss_function

__all__ = ['MLPModel', 'MultiLabelModel', 'LogMAELoss', 'get_loss_function']
import torch
from torch.utils.data import DataLoader


import os
import torch
from torch.utils.data import DataLoader

class WeightedTrainer:
    """
    加权训练方法（本文方法）
    """
    def __init__(self, config, model_T, model_C, weight_model, device='cpu'):
        self.model_T = model_T
        self.model_C = model_C
        self.weight_model = weight_model
        self.device = device
        self.optimizer_T = torch.optim.Adam(self.model_T.parameters(), lr=1e-3)
        self.optimizer_C = torch.optim.Adam(self.model_C.parameters(), lr=1e-3)
        self.optimizer_W = torch.optim.Adam(self.weight_model.parameters(), lr=1e-3)
        self.p = config.get('p_treatment', 0.5)
        self.checkpoint_path = config.get('checkpoint_path', './results/checkpoints/')

    def train_on_history(self, history_loader):
        global_epoch = 1
        for (X, Y, Z) in history_loader:
            if X.shape[0] == 1:
                print(f"[调试] 跳过 batch_size=1 (epoch={global_epoch})")
                global_epoch += 1
                continue
            X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
            print(f"[调试] epoch={global_epoch}, batch_size={X.shape[0]}")
            print(f"[调试] X 是否有 NaN: {torch.isnan(X).any().item()}, 是否有 Inf: {torch.isinf(X).any().item()}")
            # 训练权重模型 G
            self.weight_model.train()
            self.optimizer_W.zero_grad()
            pred_w = self.weight_model(X)
            loss_w = torch.nn.functional.binary_cross_entropy(pred_w, Z.float())
            loss_w.backward()
            self.optimizer_W.step()
            # 计算权重
            weight_T = pred_w / self.p
            weight_C = (1 - pred_w) / (1 - self.p)
            # 加权训练实验组模型
            self.model_T.train()
            self.optimizer_T.zero_grad()
            pred_click_t, pred_time_t = self.model_T(X)
            loss_t = self.model_T.loss_function(pred_click_t, pred_time_t, Y, weights=weight_T)
            loss_t.backward()
            self.optimizer_T.step()
            # 加权训练对照组模型
            self.model_C.train()
            self.optimizer_C.zero_grad()
            pred_click_c, pred_time_c = self.model_C(X)
            loss_c = self.model_C.loss_function(pred_click_c, pred_time_c, Y, weights=weight_C)
            loss_c.backward()
            self.optimizer_C.step()
            global_epoch += 1

    def save_models(self, step):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(self.model_T.state_dict(), os.path.join(self.checkpoint_path, f'model_T_{step}.pt'))
        torch.save(self.model_C.state_dict(), os.path.join(self.checkpoint_path, f'model_C_{step}.pt'))
        torch.save(self.weight_model.state_dict(), os.path.join(self.checkpoint_path, f'weight_model_{step}.pt'))

class PoolingTrainer:
    """
    数据池化方法：每次优化使用全部历史数据
    """
    def __init__(self, config, model, device='cpu'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.checkpoint_path = config.get('checkpoint_path', './results/checkpoints/')

    def train_on_history(self, history_loader):
        global_epoch = 1
        for (X, Y, Z) in history_loader:
            if X.shape[0] == 1:
                print(f"[调试] 跳过 batch_size=1 (epoch={global_epoch})")
                global_epoch += 1
                continue
            X, Y = X.to(self.device), Y.to(self.device)
            print(f"[调试] epoch={global_epoch}, batch_size={X.shape[0]}")
            self.model.train()
            self.optimizer.zero_grad()
            pred_click, pred_time = self.model(X)
            loss = self.model.loss_function(pred_click, pred_time, Y)
            loss.backward()
            self.optimizer.step()
            global_epoch += 1

    def train_on_pooling(self, pooling_loader):
        """专门用于预训练的pooling方法"""
        epochs = 5  # 预训练轮数
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            for (X, Y, Z) in pooling_loader:
                if X.shape[0] == 1:
                    continue
                X, Y = X.to(self.device), Y.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                pred_click, pred_time = self.model(X)
                loss = self.model.loss_function(pred_click, pred_time, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            if batch_count > 0:
                print(f"[Pretrain] Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/batch_count:.4f}")

    def save_model(self, step):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f'pooling_model_{step}.pt'))

class SplittingTrainer:
    """
    数据分割方法：实验组/对照组分开训练
    """
    def __init__(self, config, model_T, model_C, device='cpu'):
        self.model_T = model_T
        self.model_C = model_C
        self.device = device
        self.optimizer_T = torch.optim.Adam(self.model_T.parameters(), lr=1e-3)
        self.optimizer_C = torch.optim.Adam(self.model_C.parameters(), lr=1e-3)
        self.checkpoint_path = config.get('checkpoint_path', './results/checkpoints/')

    def train_on_history(self, history_loader):
        global_epoch = 1
        for (X, Y, Z) in history_loader:
            if X.shape[0] == 1:
                print(f"[调试] 跳过 batch_size=1 (epoch={global_epoch})")
                global_epoch += 1
                continue
            X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
            print(f"[调试] epoch={global_epoch}, batch_size={X.shape[0]}")
            # 实验组
            mask_T = (Z == 1)
            if mask_T.any():
                X_T, Y_T = X[mask_T], Y[mask_T]
                self.model_T.train()
                self.optimizer_T.zero_grad()
                pred_click_t, pred_time_t = self.model_T(X_T)
                loss_t = self.model_T.loss_function(pred_click_t, pred_time_t, Y_T)
                loss_t.backward()
                self.optimizer_T.step()
            # 对照组
            mask_C = (Z == 0)
            if mask_C.any():
                X_C, Y_C = X[mask_C], Y[mask_C]
                self.model_C.train()
                self.optimizer_C.zero_grad()
                pred_click_c, pred_time_c = self.model_C(X_C)
                loss_c = self.model_C.loss_function(pred_click_c, pred_time_c, Y_C)
                loss_c.backward()
                self.optimizer_C.step()
            global_epoch += 1

    def save_models(self, step):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(self.model_T.state_dict(), os.path.join(self.checkpoint_path, f'split_model_T_{step}.pt'))
        torch.save(self.model_C.state_dict(), os.path.join(self.checkpoint_path, f'split_model_C_{step}.pt'))

class SnapshotTrainer:
    """
    快照法：用初始数据拟合模型，后续不再更新
    """
    def __init__(self, config, model, device='cpu'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.checkpoint_path = config.get('checkpoint_path', './results/checkpoints/')

    def train_on_snapshot(self, snapshot_loader):
        global_epoch = 1
        for (X, Y, Z) in snapshot_loader:
            if X.shape[0] == 1:
                print(f"[调试] 跳过 batch_size=1 (epoch={global_epoch})")
                global_epoch += 1
                continue
            X, Y = X.to(self.device), Y.to(self.device)
            print(f"[调试] epoch={global_epoch}, batch_size={X.shape[0]}")
            self.model.train()
            self.optimizer.zero_grad()
            pred_click, pred_time = self.model(X)
            loss = self.model.loss_function(pred_click, pred_time, Y)
            loss.backward()
            self.optimizer.step()
            global_epoch += 1

    def save_model(self, step):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f'snapshot_model_{step}.pt'))


import logging
import os
import torch


def setup_logging(config):
    log_path = config.get('log_path', './results/logs/')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_path, 'experiment.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    return logging.getLogger()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def compute_metrics(y_true, y_pred):
    """
    计算常用指标：点击率、均方误差、长播率等
    y_true, y_pred: [N, 2] (is_click, play_time_ms)
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    click_true = y_true[:, 0]
    click_pred = torch.sigmoid(y_pred[:, 0])
    play_true = y_true[:, 1]
    play_pred = y_pred[:, 1]
    ctr = click_true.mean().item()
    ctr_pred = click_pred.mean().item()
    mse = torch.mean((play_true - play_pred) ** 2).item()
    long_view_true = (play_true > 10000).float().mean().item() # 10秒阈值
    long_view_pred = (play_pred > 10000).float().mean().item()
    return {
        'CTR_true': ctr,
        'CTR_pred': ctr_pred,
        'MSE_play_time': mse,
        'LongView_true': long_view_true,
        'LongView_pred': long_view_pred,
        'play_time_true': play_true.mean().item(),
        'play_time_pred': play_pred.mean().item()
    }

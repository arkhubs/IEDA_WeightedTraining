
import json
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) > 1:
    result_dir = sys.argv[1]
else:
    result_dir = os.path.dirname(__file__)
json_path = os.path.join(result_dir, 'exp_results.json')
if not os.path.exists(json_path):
    raise FileNotFoundError(f"未找到结果文件: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

steps = data['step']
metrics = data['metrics']

# 新结构：metrics为{'train':..., 'val':...}
def extract_curve(metric_key):
    tr = [m['train'][metric_key] if m['train'] and metric_key in m['train'] else None for m in metrics]
    val = [m['val'][metric_key] if m['val'] and m['val'] and metric_key in m['val'] else None for m in metrics]
    return tr, val

auc_tr, auc_val = extract_curve('CTR_AUC')
logmae_tr, logmae_val = extract_curve('LogMAE_play_time')

plt.figure(figsize=(12, 5))

# 1. CTR AUC 曲线
plt.subplot(1,2,1)
plt.plot(steps, auc_tr, label='Train CTR AUC', marker='o')
plt.plot(steps, auc_val, label='Val CTR AUC', marker='x')
plt.xlabel('Step')
plt.ylabel('AUC')
plt.title('CTR ROC AUC')
plt.legend()

# 2. playtime Log-MAE 曲线
plt.subplot(1,2,2)
plt.plot(steps, logmae_tr, label='Train Log-MAE', marker='o')
plt.plot(steps, logmae_val, label='Val Log-MAE', marker='x')
plt.xlabel('Step')
plt.ylabel('Log-MAE')
plt.title('Playtime Log-MAE')
plt.legend()

plt.tight_layout()
save_path = os.path.join(result_dir, 'exp_results_plot.png')
plt.savefig(save_path, dpi=200)
print(f"图像已保存到: {save_path}")
plt.show()

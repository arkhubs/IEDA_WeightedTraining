import json
import matplotlib.pyplot as plt
import os

# 读取结果文件
dir_path = os.path.dirname(__file__)
json_path = os.path.join(dir_path, 'exp_results.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

steps = data['step']
metrics = data['metrics']

# 提取各项指标
CTR_true = [m['CTR_true'] for m in metrics]
CTR_pred = [m['CTR_pred'] for m in metrics]
MSE_play_time = [m['MSE_play_time'] for m in metrics]
LongView_true = [m['LongView_true'] for m in metrics]
LongView_pred = [m['LongView_pred'] for m in metrics]
play_time_true = [m.get('play_time_true', None) for m in metrics]
play_time_pred = [m.get('play_time_pred', None) for m in metrics]

# 绘图
plt.figure(figsize=(14, 10))

plt.subplot(2,2,1)
plt.plot(steps, CTR_true, label='CTR_true')
plt.plot(steps, CTR_pred, label='CTR_pred')
plt.xlabel('Step')
plt.ylabel('CTR')
plt.title('CTR True vs Pred')
plt.legend()

plt.subplot(2,2,2)
plt.plot(steps, MSE_play_time, label='MSE_play_time')
plt.xlabel('Step')
plt.ylabel('MSE')
plt.title('MSE of Play Time')
plt.legend()

plt.subplot(2,2,3)
plt.plot(steps, LongView_true, label='LongView_true')
plt.plot(steps, LongView_pred, label='LongView_pred')
plt.xlabel('Step')
plt.ylabel('LongView Ratio')
plt.title('LongView True vs Pred')
plt.legend()

plt.subplot(2,2,4)
if all(v is not None for v in play_time_true):
    plt.plot(steps, play_time_true, label='play_time_true')
if all(v is not None for v in play_time_pred):
    plt.plot(steps, play_time_pred, label='play_time_pred')
plt.xlabel('Step')
plt.ylabel('Play Time')
plt.title('Play Time True vs Pred')
plt.legend()

plt.tight_layout()
save_path = os.path.join(dir_path, 'exp_results_plot.png')
plt.savefig(save_path, dpi=200)
print(f"图像已保存到: {save_path}")
plt.show()

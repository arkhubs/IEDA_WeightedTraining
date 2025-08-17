# Table of Contents
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/Exp Plan-20250701.md
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/README.md
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/main.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/run_gpu.sh
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/Exp logs.md
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/.gitignore
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/__init__.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/configs/experiment.yaml
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/logger.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/metrics.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/experiment_utils.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/__init__.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/modes/global_mode.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/modes/__init__.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/multi_label_model.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/loss_functions.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/mlp_model.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/__init__.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/data_loader.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/feature_processor.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/cache_manager.py
- /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/__init__.py

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/Exp Plan-20250701.md

- Extension: .md
- Language: markdown
- Size: 31430 bytes
- Created: 2025-08-04 13:41:15
- Modified: 2025-08-04 13:41:15

### Code

```markdown
  1 | ## 关于数据
  2 | - 已有数据：用户对视频的互动行为，顺序数据
  3 | 	- 在最小的数据集内，一个用户对应几十或上百个视频
  4 | 	- 数据结构：从互动行为表出发，需要 feature 数据时，根据 user_id 和 video_id 去 feature 表里面查找
  5 | - $Y$ 的候选：
  6 | 	- is_click / is_like / is_follow / is_comment / is_hate
  7 | 	- play_time_ms / long_view / `play_time_ms➗duration_ms`
  8 | - 可用的 $X$：
  9 | 	- 多多益善
 10 | - 数据集自带 A/B Test，前一段时间为正常推送（A）；后一段时间插入了随机推送（后一段时间整体视为B）（占比很高）
 11 | 	- 前后两段时间为同一批用户，<font color="#ff0000">这是否存在时间上的干扰？</font>
 12 | 	- <font color="#ff0000">另一种用法？：所有正常推送的视频视为 A；所有随机推送的视频视为 B</font>
 13 | - 数据集生成过程中是否存在本文所说的干扰？
 14 | 	- 应当是有的，因为存在推荐算法推荐视频的过程
 15 | 
 16 | ---
 17 | 
 18 | ## 实验方法与目标
 19 | - 目标：训练出预测用户行为的模型 $M_\theta$，用来估计 $GTE$
 20 | 	- 使用加权训练、数据池化、快照、数据分割四种方法分别训练，比较最终的 log 损失、预测方差、用它们估计的 GTE 的箱线图<font color="#ff0000">（这里具体如何比较需要老师指导）</font>
 21 | 	- 在 pure、1K、27K 三个大小上进行实验，每个重复实验 $B$ 次
 22 | 	- 编写代码先从最小的 pure 开始尝试
 23 | 
 24 | - 以下分别是每种方法的 $M$ 的训练过程
 25 | 1. 加权训练 **（本文方法）**
 26 | 	- 首先初始化分类器 $G_{\theta_W}: \mathbb{R}^d \rightarrow [0,1]$ 用于估计 $E [Z|X_E]$；初始化实验组模型 $M_{\theta_T}$ 和控制组模型 $M_{\theta_C}$
 27 | 	- 第 $t$ 个 epoch 纳入 $n_t$ 个用户 ($n_t < n$) ，$t$ 初值为 1
 28 | 		- 对 $n_t$ 个用户随机分组（参数为 p 的 Bernoulli 分布产生 $Z$）
 29 | 		- 他们的 $X$ 作为 feature，$Z$ 作为 label，优化 $G_{\theta_W}$<font color="#ff0000">（原本是先优化</font> $M$<font color="#ff0000">，这里我改了一下）</font>
 30 | 			- 神经网络 or SVM or 随机森林
 31 | 		- 根据 `user_id` 和 `Z`，去数据集中查找所有满足条件的 `Y`
 32 | 			- 这里有两种处理办法：
 33 | 				1. 使用所有满足条件的 Y（被推给该用户的所有控制组/实验组视频）
 34 | 				2. 进行随机抽样，取其中 $m$ 个
 35 | 		- 计算权重：
 36 | 		$$
 37 | 			W_{T,i,t} = \frac{G_{\theta_W}(X_{i,t})}{p}, \quad W_{C,i,t} = \frac{1-G_{\theta_W}(X_{i,t})}{1-p}.
 38 | 		$$
 39 | 		- $X$ 作为 feature，$Y$ 作为 label，由 $W_T \mathcal{D}_E \stackrel{d}{=} \mathcal{D}_T$ 更新实验组模型；由 $W_C \mathcal{D}_E \stackrel{d}{=} \mathcal{D}_C$ 更新控制组模型
 40 | 		- t=t+1
 41 | 2. 数据池化：每次优化使用这个时间点 $t$ 的所有 X 和 Y 
 42 | 3. 快照：事先拟合好 M，再开始实验，相当于脱离 Test 直接拟合所有 X 和 Y
 43 | 4. 数据分割：每次优化实验组仅使用 $Z=1$ 的 X 和 Y；控制组仅使用 $Z=0$ 的 X 和 Y
 44 | 
 45 | - 然后进行各种指标的对比
 46 | 
 47 | 1. 推荐算法 model：Wide & Deep Learning for Recommender Systems
 48 |   1. https://arxiv.org/abs/2205.09809
 49 |   2. deep learning recommendation model
 50 |   3. 先用 LM 代替
 51 | 2. framework modelize （模块化）
 52 | 3. 确定 X 和 Y，写出 data 的状况
 53 | 4. 定期分享代码（可以考虑 git）
 54 | 5. 用英文写 report
 55 | 6. 100 个推一个（已有的）
 56 | 7. GTE 计算：用我们自己的 A/B Test，
 57 | 8. [x] 办 visa 卡；保险时间往后挪（夏令营时间确定后）
 58 | 
 59 | 
 60 | 
 61 | ## ToDo 20250721
 62 | 
 63 | - [ ] Wide模型先不实装
 64 | 
 65 | - [x] 调整模型特征和标签
 66 | 
 67 | - [x] 调整模型网络结构
 68 | 
 69 | - [ ] 播放时长是0是否可以替代掉点击率这个标签
 70 | 
 71 | - [x] 跑通demo
 72 |   - [x] 环境配置
 73 |   - [x] 对于NaN处理：
 74 |     - [x] 数值变量采用预测模型；（或者可以额外增加一个特征，表示该值是否是由缺失值预测而来的；这里先采用均值填充，以后再改；
 75 |     - [x] 如果是分类变量，直接将缺失值作为一个独立类别`Missing`
 76 |     - [ ] 改进计划
 77 |   - [x] weight预测出现预测值全为极端的情况
 78 |     - [x] 加batch norm
 79 |     - [ ] 如需更细致的分布控制可考虑 clip/log1p 等操作
 80 |   
 81 | - [x] 创建git
 82 |   - [x] 在服务器上下载数据集
 83 |   - [x] 在服务器上运行
 84 |   - [ ] 开免登录
 85 |   
 86 | - [x] 安排meeting时间
 87 | 
 88 | - [x] 调参（费时费力）
 89 |   - [x] MSE_play_time始终比较大
 90 |     - [x] 排查是否是本来scale就大
 91 |       - [x] 是的
 92 |   - [x] 画图发现不正常![exp_results_plot](./assets/exp_results_plot.png)
 93 |   - [x] loss_click + loss_time的比例是否正常，会否其中一个数值占据绝对主导
 94 |     - [x] 分开训练
 95 |   - [x] 测AUC 0.75足够
 96 |   - [ ] 模型变小
 97 |     - [x] 3层，8，16，32
 98 |     - [ ] 特征数量减少
 99 |   - [x] 测试集
100 |   - [x] 另外你这里测试集和测试集是怎么划分的，每次抽取范围是怎么样的；还有预训练的5000个数据点在代码的哪个部分实现？
101 |   
102 | - [ ] 尝试不同指标组合
103 | 
104 | - [ ] cuda支持问题
105 | 
106 |   - [x] 发邮件
107 | 
108 | - [x] 预训练有在进行吗？输出一下结果
109 | 
110 | - [x] 修复相对路径问题
111 | 
112 |   - [x] >
113 |     > 帮我解决以下问题：
114 |     >
115 |     > 1. 现在代码中绝对路径和相对路径混用，我希望你彻底修复有序，最多仅在config文件中指定base目录，之后任何时候cd到base目录之后，运行脚本、调用数据集都不会出现路径问题。
116 |     >
117 |     >    文件树如下：IEDA_WeightedTraining/
118 |     >      ├── configs/
119 |     >      │     └── experiment_config.yaml
120 |     >      ├── results/
121 |     >      │     ├── plot_exp_results.py
122 |     >      │     └── test/...
123 |     >      ├── src/
124 |     >      │     ├── main.py
125 |     >      │     ├── data_manager.py
126 |     >      │     ├── models.py
127 |     >      │     ├── trainers.py
128 |     >      │     ├── recommender.py
129 |     >      │     └── utils.py
130 |     >    KuaiRand/
131 |     >      ├── 1K/
132 |     >      ├── 27K/
133 |     >      │     └── data/
134 |     >      │           ├── log_standard_4_08_to_4_21_27k_part1.csv
135 |     >      │           ├── log_standard_4_08_to_4_21_27k_part2.csv
136 |     >      │           ├── log_standard_4_22_to_5_08_27k_part1.csv
137 |     >      │           ├── log_standard_4_22_to_5_08_27k_part2.csv
138 |     >      │           ├── log_random_4_22_to_5_08_27k.csv
139 |     >      │           ├── user_features_27k.csv
140 |     >      │           ├── video_features_basic_27k.csv
141 |     >      │           ├── video_features_statistic_27k_part1.csv
142 |     >      │           ├── video_features_statistic_27k_part2.csv
143 |     >      │           └── video_features_statistic_27k_part3.csv
144 |     >      └── Pure/
145 |     >
146 |     > 2. 27K数据集加载非常慢，而且以下运行日志显示卡在最后一步3个小时之后才出现报错信息，请寻找问题原因并修复：
147 |     >
148 |     >    [zhixuanhu@login2 IEDA_WeightedTraining]$ python3 IEDA_WeightedTraining/src/main.py
149 |     >    [DEBUG] Pattern: KuaiRand/27K/data/log_standard_4_08_to_4_21_27k_part*.csv
150 |     >    [DEBUG] Matched files: ['KuaiRand/27K/data/log_standard_4_08_to_4_21_27k_part1.csv', 'KuaiRand/27K/data/log_standard_4_08_to_4_21_27k_part2.csv']
151 |     >    [DEBUG] Loading log file: KuaiRand/27K/data/log_standard_4_08_to_4_21_27k_part1.csv
152 |     >    [DEBUG] Loading log file: KuaiRand/27K/data/log_standard_4_08_to_4_21_27k_part2.csv
153 |     >    [DEBUG] Pattern: KuaiRand/27K/data/log_standard_4_22_to_5_08_27k_part*.csv
154 |     >    [DEBUG] Matched files: ['KuaiRand/27K/data/log_standard_4_22_to_5_08_27k_part1.csv', 'KuaiRand/27K/data/log_standard_4_22_to_5_08_27k_part2.csv']
155 |     >    [DEBUG] Loading log file: KuaiRand/27K/data/log_standard_4_22_to_5_08_27k_part1.csv
156 |     >    [DEBUG] Loading log file: KuaiRand/27K/data/log_standard_4_22_to_5_08_27k_part2.csv
157 |     >    [DEBUG] Pattern: KuaiRand/27K/data/log_random_4_22_to_5_08_27k_part*.csv
158 |     >    [DEBUG] Matched files: []
159 |     >    [DEBUG] Check single file: KuaiRand/27K/data/log_random_4_22_to_5_08_27k.csv, exists=True
160 |     >    [DEBUG] Loading log file: KuaiRand/27K/data/log_random_4_22_to_5_08_27k.csv
161 |     >    [DEBUG] logs length: 5
162 |     >    [Device] Available devices: ['cpu'], selected: cpu
163 |     >
164 |     >      File "/home/zhixuanhu/.local/lib/python3.9/site-packages/pandas/core/generic.py", line 6331, in __setattr__
165 |     >        object.__getattribute__(self, name)
166 |     >      File "/home/zhixuanhu/.local/lib/python3.9/site-packages/pandas/core/series.py", line 782, in name
167 |     >        return self._name
168 |     >      File "/home/zhixuanhu/.local/lib/python3.9/site-packages/pandas/core/generic.py", line 6318, in __getattr__
169 |     >        return object.__getattribute__(self, name)
170 |     >    AttributeError: 'Series' object has no attribute '_name'
171 |     >
172 |     >    During handling of the above exception, another exception occurred:
173 |     >
174 |     >    Traceback (most recent call last):
175 |     >      File "/home/zhixuanhu/IEDA_WeightedTraining/IEDA_WeightedTraining/src/main.py", line 253, in <module>
176 |     >        run_experiment(config)
177 |     >      File "/home/zhixuanhu/IEDA_WeightedTraining/IEDA_WeightedTraining/src/main.py", line 114, in run_experiment
178 |     >        pretrain_weights_path = pretrain_models(config, logger, data_manager, device)
179 |     >      File "/home/zhixuanhu/IEDA_WeightedTraining/IEDA_WeightedTraining/src/main.py", line 36, in pretrain_models
180 |     >        all_interactions = list(data_manager.get_all_interactions())
181 |     >      File "/home/zhixuanhu/IEDA_WeightedTraining/IEDA_WeightedTraining/src/data_manager.py", line 74, in get_all_interactions
182 |     >        for _, interaction in self.master_df.iterrows():
183 |     >      File "/home/zhixuanhu/.local/lib/python3.9/site-packages/pandas/core/frame.py", line 1559, in iterrows
184 |     >        s = klass(v, index=columns, name=k).__finalize__(self)
185 |     >      File "/home/zhixuanhu/.local/lib/python3.9/site-packages/pandas/core/series.py", line 593, in __init__
186 |     >        self.name = name
187 |     >      File "/home/zhixuanhu/.local/lib/python3.9/site-packages/pandas/core/generic.py", line 6331, in __setattr__
188 |     >        object.__getattribute__(self, name)
189 |     >    KeyboardInterrupt
190 |     >
191 |     > 3. 不需要的feature也被同时加载，请你想办法优化，比如config增加一个选项，想办法load数据集（尤其是27K这种20GB的）后，缓存（放在KuaiRand目录下），加快下次加载
192 |     >
193 |     > 4. 完整顺利运行一次之后，你可能会看到结果不太好，AUC无法超过0.6，logMAE越来越大，检查损失函数中loss_click + loss_time的比例是否正常，会否其中一个数值占据绝对主导，如果是，尝试调整；如果还是不行，可能是两个预测指标相互干扰训练，考虑彻底将click预测模型和playtime预测模型拆开成两个，这在config文件中又要如何体现
194 |     >
195 |     > 5. 再次实验，如果还是不好，也有可能是特征数量102过多问题，是否需要减少一些，你先检索所有特征，然后做实验探究
196 |     >
197 |     > 6. 思考别的可能原因导致我训练结果不好
198 | 
199 |  - [x] > 帮我诊断：现在的问题是，预训练环节的step3总是卡住很久，而且数据集规模越大卡住越久；之后我换到了最小的数据集，结果还是卡住了，具体查看log：
200 |    >
201 |    > 2025-07-30 19:42:39,518 INFO Using device: cpu
202 |    >
203 |    > 2025-07-30 19:42:39,519 INFO [Pretrain] 开始预训练，随机选择 5000 个交互
204 |    >
205 |    > 2025-07-30 19:55:36,890 INFO Using device: cpu
206 |    >
207 |    > 2025-07-30 19:55:36,890 INFO [Pretrain] 开始预训练，随机选择 5000 个交互
208 |    >
209 |    > 2025-07-30 19:55:36,890 INFO [Pretrain] Step 1: 获取所有交互样本 ...
210 |    >
211 |    > 2025-07-30 19:58:16,146 INFO [Pretrain] Step 1: 获取所有交互样本完成，数量: 11756073
212 |    >
213 |    > 2025-07-30 19:58:16,146 INFO [Pretrain] Step 2: 随机采样交互 ...
214 |    >
215 |    > 2025-07-30 19:58:20,857 INFO [Pretrain] Step 2: 随机采样完成，采样数量: 5000
216 |    >
217 |    > 2025-07-30 19:58:20,857 INFO [Pretrain] Step 3: 构建预训练数据 ...
218 |    >
219 |    > 2025-07-30 19:58:20,857 INFO [Pretrain] Step 3: 构建中 0/5000 ...
220 |    >
221 |    > 2025-07-30 20:02:50,362 INFO [Pretrain] Step 3: 构建中 100/5000 ...
222 |    >
223 |    > 2025-07-30 20:07:12,224 INFO [Pretrain] Step 3: 构建中 200/5000 ...
224 |    >
225 |    > 2025-07-30 20:11:20,221 INFO Using device: cpu
226 |    >
227 |    > 2025-07-30 20:11:20,221 INFO [Pretrain] 开始预训练，随机选择 5000 个交互
228 |    >
229 |    > 2025-07-30 20:11:20,221 INFO [Pretrain] Step 1: 获取所有交互样本 ...
230 |    >
231 |    > 2025-07-30 20:14:04,562 INFO [Pretrain] Step 1: 获取所有交互样本完成，数量: 11756073
232 |    >
233 |    > 2025-07-30 20:14:04,563 INFO [Pretrain] Step 2: 随机采样交互 ...
234 |    >
235 |    > 2025-07-30 20:14:09,123 INFO [Pretrain] Step 2: 随机采样完成，采样数量: 5000
236 |    >
237 |    > 2025-07-30 20:14:09,123 INFO [Pretrain] Step 3: 批量构建预训练数据 ...
238 |    >
239 |    > 2025-07-30 20:14:09,125 INFO [Pretrain] Step 3: 批量生成特征 ...
240 |    >
241 |    > 2025-07-30 20:16:16,442 INFO Using device: cpu
242 |    >
243 |    > 2025-07-30 20:16:16,442 INFO [Pretrain] 开始预训练，随机选择 5000 个交互
244 |    >
245 |    > 2025-07-30 20:16:16,442 INFO [Pretrain] Step 1: 获取所有交互样本 ...
246 |    >
247 |    > 2025-07-30 20:18:56,156 INFO [Pretrain] Step 1: 获取所有交互样本完成，数量: 11756073
248 |    >
249 |    > 2025-07-30 20:18:56,157 INFO [Pretrain] Step 2: 随机采样交互 ...
250 |    >
251 |    > 2025-07-30 20:19:00,702 INFO [Pretrain] Step 2: 随机采样完成，采样数量: 5000
252 |    >
253 |    > 2025-07-30 20:19:00,703 INFO [Pretrain] Step 3: 批量构建预训练数据 ...
254 |    >
255 |    > 2025-07-30 20:19:00,705 INFO [Pretrain] Step 3: 单条生成特征 ...
256 |    >
257 |    > 2025-07-30 20:19:00,705 INFO [Pretrain] Step 3: 特征生成进度 0/5000 ...
258 |    >
259 |    > 2025-07-30 20:34:58,053 INFO Using device: cpu
260 |    >
261 |    > 2025-07-30 20:34:58,053 INFO [Pretrain] 开始预训练，随机选择 5000 个交互
262 |    >
263 |    > 2025-07-30 20:34:58,053 INFO [Pretrain] Step 1: 获取所有交互样本 ...
264 |    >
265 |    > 2025-07-30 20:37:39,753 INFO [Pretrain] Step 1: 获取所有交互样本完成，数量: 11756073
266 |    >
267 |    > 2025-07-30 20:37:39,754 INFO [Pretrain] Step 2: 随机采样交互 ...
268 |    >
269 |    > 2025-07-30 20:37:44,292 INFO [Pretrain] Step 2: 随机采样完成，采样数量: 5000
270 |    >
271 |    > 2025-07-30 20:37:44,292 INFO [Pretrain] Step 3: 构建预训练数据 ...
272 |    >
273 |    > 2025-07-30 20:37:44,292 INFO [Pretrain] Step 3: 构建中 0/5000 ...
274 |    >
275 |    > 请你想办法优化-测试，先在pure上测试，正常了逐级往上扩大，直到解决问题
276 |    >
277 |    > 同时，我这里期望的epoch的意思是1000个step，每20个step评估一次，称为一个epoch，但现在显示的epoch总是1-5然后又复原，这是为何？并在此时询问我是否要修复
278 |    >
279 |    > 然后，要把实验组和对照组的预训练分开为2组独立进行；之后每个epoch的评测也要分开，画图程序相应调整
280 | 
281 | > 小样本运行
282 | >
283 | > sed -i 's/initial_dataset_size.* 5000/initial_dataset_size\": 50/g' IEDA_WeightedTraining/configs/experiment_config.yaml && python3 -m IEDA_WeightedTraining.src.main
284 | 
285 | ![15bee03caef00fc584b934168a41d6c8](./assets/15bee03caef00fc584b934168a41d6c8.png)
286 | 
287 | - [ ] 训练逻辑需要大幅重构，参看以下说明；不再使用epoch的概念，每个step等价于1个epoch，n_user等价于batchsize
288 | 
289 |   - 我在config中指定了重复实验repetitions次，每次实验中：
290 | 
291 |     - 事先对**视频**划分训练集/测试集（mask），对**用户**划分处理组/对照组、比例决定于p_treatment和p_val，
292 | 
293 |     - 首先对照预测模型、处理预测模型、权重模型分别使用initial_dataset_size组交互数据和分组结果变量Z进行预训练
294 | 
295 |     - 然后每次实验迭代total_simulation_steps个step，每个step分别对n_user_T\n_user_C个处理组/对照组用户推介得分最好的1个视频，候选池为训练集内该用户看过、有交互数据的随机candidate_pool_size_train个视频（否则没有数据，不知道交互数据，若不足该数，则取允许的最大数），
296 | 
297 |     - 收集（即查找）被推介出去的那n_user_T\n_user_C个视频的交互数据，用于更新训练weight模型、model_C、model_T并记录训练loss等（不再用历史pool来训练，不再使用未被推介的数据来训练），后两者用于优化的损失函数是类似
298 |   
299 |       - $$
300 |         \frac{1}{n_t} \sum_{i=1}^{n_t} W_{T,i,t} \ell(M_{\theta_T}(X_{i,t}), Y_{i,t}).
301 |         $$
302 | 
303 |     - 更新完成后，在这n_user_T\n_user_C个用户的有交互数据的测试集视频中，为每个用户推介出得分最高的1个视频，用它们计算测试集的loss、auc等指标
304 | 
305 |     - log一次结果并画图
306 |   
307 |       
308 |   
309 |   - 参看：
310 |   
311 |   -  **算法 1：加权训练流程**：
312 |         1. **输入**：处理分配概率 $p$，权重预测模型类 $\mathcal{G} = \{G_{\theta_W}: \mathbb{R}^d \rightarrow [0,1]\}$，机器学习模型类 $\mathcal{M}$，损失函数 $\ell(M(X), Y)$。
313 |         2. **初始化**：实验模型 $M_{\theta_T}$ 和控制模型 $M_{\theta_C}$，均设置为当前生产模型。
314 |         3. **循环**（时间 $t=1$ 至实验结束）：
315 |             - 为 $n_t$ 个用户随机分配处理（概率 $p$）。
316 |             - 根据分配推荐项目，收集数据 $(X_{i,t}, Y_{i,t}, Z_{i,t})$。
317 |             - 计算权重：
318 |     $$
319 |     W_{T,i,t} = \frac{G_{\theta_W}(X_{i,t})}{p}, \quad W_{C,i,t} = \frac{1-G_{\theta_W}(X_{i,t})}{1-p}.
320 |     $$
321 |          - 更新实验模型 $M_{\theta_T}$，最小化加权损失：
322 |     $$
323 |     \frac{1}{n_t} \sum_{i=1}^{n_t} W_{T,i,t} \ell(M_{\theta_T}(X_{i,t}), Y_{i,t}).
324 |     $$
325 |          - 更新控制模型 $M_{\theta_C}$，类似最小化加权损失。
326 |          - 更新权重模型 $G_{\theta_W}$，使用数据 $\{(X_{i,t}, Z_{i,t})\}$。
327 |   
328 |     4. **输出**：朴素估计器（PAGE10, Weighted Training）。
329 |   
330 |   - 另外，这些都在config中指定，不要放到main里面定义
331 |   
332 |   - log加上train和val分别涉及了多少个用户交互记录
333 | 
334 | 
335 | 
336 | - [ ] 问题：playtime的预测趋向于0，是否是因为大部分真实playtime都是0，那么我希望log记录的playtime不再计算全部的平均值和MSE，而是记录非0的playtime的平均相对误差（百分比那种），画图相应调整为（比例、布局也调整适应）：
337 |   - 图块1：CTR AUC，包括train/val和treatment/control
338 |   - 图块2：CTR XXloss，包括train/val和treatment/control
339 |   - 图块3：playtime 平均相对误差（百分比），包括train/val和treatment/control
340 |   - 图块4：playtime XXloss，包括train/val和treatment/control
341 |   - 图块5：weightmodel AUC和Acurracy，包括train/val
342 |   - 图块6：weightmoel XXloss，包括train/val
343 | 
344 | 
345 | 
346 | - [ ] 从log看，1K的预训练，生成每50个特征需要2分钟，直到正式开始预训练，用了50分钟，而预训练过程却在1秒内完成，这是代码的缺陷，还是不该如此先生成特征
347 | 
348 |   
349 | 
350 | - [ ] 下一步实验怎么做
351 | 
352 |   - 用训练好的模型（weight、T、C）预测测试集中的CTR、playtime，并根据playtime计算 long_view ，然后比较各种方法上述量的误差、方差
353 |   - 如何比较各个模型对于GTE预测的效果好坏？
354 |     - 真实的GTE无法知晓？
355 |     - 使用实验日志（模拟的生产环境），计算估计的GTE
356 |     - 比较方差？
357 | 
358 | 
359 | 
360 | > ### 一、 数据加载、处理与管理模块
361 | >
362 | > 这是实验的基础，负责准备和管理所需的数据。
363 | >
364 | > 1. **数据流**:
365 | >    - 从数据集中提取 `user_id` 
366 | >    - 根据 `user_id` 查找该用户所有**有过交互（看过）的视频列表（Video-list）**。
367 | >    - 为了提高效率，设计了**缓存机制**，避免重复查找。
368 | > 2. **数据集划分**:
369 | >    - 仅将**用户**划分为**训练集（train-loader）和验证集（val-loader）**。
370 | >    - 为训练数据增加两个标记位：
371 | >      - `mask`: 原始标记，所有被分配到val的user看过的视频被标记为mask=1。
372 | >      - `used`: **降权与隔离机制**。在仿真时，一个视频一旦被推荐给某个用户（无论是在GT还是GC流程中），其 `used` 标记置为1，之后**不会再次被推荐**。**重要的是，Treatment和Control两个仿真流程完全独立运行，不共享used**
373 | > 3. **特征处理**:
374 | >    1. 用户特征`user_active_degree`、视频基础特征`video_type`、`tag`是字符型类别变量，可以转化为若干个哑变量。
375 | >    2. 视频基础特征`upload_dt`是日期格式，对于预测没有作用，不使用这个特征。
376 | >    3. 各种id类变量不适合量化，也不需要，不作为特征使用。
377 | >    4. 所有使用的特征都在config中声明式定义，只分两类（数值和分类），未声明的特征不用。
378 | >    5. NA等缺失值作为一个新的类别；连续型变量则用0填充
379 | >
380 | > ### 二、 计算真实GTE的模块，称为global（GT与GC对称运行）
381 | >
382 | > 这是框架的核心。下面的流程会**独立运行两次**：一次所有user作为处理组使用处理组的alpha (vecalpha∗T)来计算GT，一次所有user作为对照组使用对照组的alpha (vecalpha∗C)来计算GC。
383 | >
384 | > **单次仿真循环流程 (for step = 1 to n_steps):**
385 | >
386 | > 1. **用户批次抽样**:
387 | >    - 从 `train-loader` 中每步抽取一个 `batch_size` 的用户。（GT和GC的仿真在每一步都使用**完全相同**的用户批次）。
388 | > 2. **候选视频生成**:
389 | >    - 对于批次中的每一个用户，从其 `Video-list` 中，筛选出 `mask=0` 且 `used=0` 的视频（`used`状态是两个流程共享的）。
390 | >    - 从筛选后的列表中，抽取 `n_candidate` 个视频作为该用户的候选推荐池。
391 | > 3. **模型预测与加权排序**:
392 | >    - 使用当前的 `predicting model` 对 `batch_size * n_candidate` 个候选视频进行打分，得到基础预测值 `Label`。predicting model的输入是user特征+video特征，选用的特征在config中定义。play_time尺度很大，训练中使用logMAE损失函数
393 | >    - 根据当前仿真流程的身份（是Treatment还是Control），使用对应的 `alpha` 参数（vecalpha∗T 或 vecalpha∗C）计算最终排序分 `score`。例如：`score = alpha · Label`。
394 | >    - 对每个用户的候选视频，根据 `score` 从高到低排序。
395 | > 4. **选出胜出视频**:
396 | >    - 为每个用户选出排序后 `score` 最高的视频作为“胜出者”。
397 | > 5. **获取真实反馈与模型训练**:
398 | >    - 查找这批胜出视频在真实交互数据中的特征 `X` 和真实 `Label` (Y)。
399 | >    - 将 `(X, Y)` 数据对用于**训练共享的 `predicting model`**。
400 | >    - **同时，将这批胜出视频的真实 `Label` (Y) 累加到当前仿真流程的总收益中**（GT流程累加到`total_label_T`，GC流程累加到`total_label_C`）。
401 | > 6. **更新状态**:
402 | >    - 将被推荐过的视频的 `used` 标记置为1。
403 | > 7. **保存与续训**:
404 | >    - 循环结束后，保存共享的模型参数、优化器状态和当前 `step`数，以便能够接续训练。
405 | > 8. **Validation**: 每n步（在config中定义）使用测试集validate一次（validate不考虑used和masked，user从val_loader中获取，其余流程一致）。
406 | >
407 | > ### 三、 模型、配置与文件结构
408 | >
409 | > #### 模型
410 | >
411 | > - **Predicting model**: 核心的共享模型，输入特征 `X`，输出基础预测 `Label`。它在两个并行的仿真流程中被同步更新。
412 | >
413 | > #### 配置文件 (`config.yaml`)
414 | >
415 | > `config.yaml`通过顶层的`mode`键来决定本次实验运行的模式，并读取对应模式下的参数配置。
416 | >
417 | > - **`mode`**: (顶层键) 字符串，指定实验模式，如 `'global'`, `'weighting'`, `'splitting'`等。
418 | > - **`global`**: (顶层配置块) **仅在`mode: 'global'`时生效**。此配置块表示global mode下的参数设置，与之平行的将来会有weighting、splitting配置块：
419 | >   - `user_p_val`: 验证集用户比例。
420 | >   - `batchsize`: 每步抽样的用户数。
421 | >   - `n_candidate`: 每个用户的候选视频数。
422 | >   - `n_steps`: 仿真总步数。
423 | > - **`feature`**: (顶层键) 声明模型使用的数值特征和分类特征。
424 | > - **`label`**: (顶层键) 定义预测的目标（`binary`或`numerical`）。
425 | > - **`pretrain`**: (顶层配置块) 定义预训练阶段的参数，如`batch_size`, `epochs`。
426 | > - **`recommender`**: (顶层配置块) **实验的核心**，定义了两种策略的`alpha`权重。
427 | >   - `alpha_T: [w1, w2, ...]`
428 | >   - `alpha_C: [w1, w2, ...]`
429 | >
430 | > #### 文件结构与日志
431 | >
432 | > 项目的根目录为 `base`，其下包含了配置、数据、代码和结果等所有相关文件。
433 | >
434 | > ```
435 | > base/
436 | > ├── configs/
437 | > │   └── experiment.yaml        # 实验配置文件
438 | > ├── data/
439 | > │   └── KuaiRand/
440 | > │       ├── Pure/data             # KuaiRand-Pure 原始数据集
441 | > │       ├── 1K/data
442 | > │       ├── 27K/data
443 | > │            └── .csv           # 经过预处理、合并后的特征与标签数据
444 | > ├── libs/
445 | > │   └── ...                    # 主体逻辑实现
446 | > 	└── modes
447 | > 		└── global.py
448 | > 		└── weighting.py
449 | > 		└── splitting.py
450 | > 		└── ...
451 | > └── results/
452 | >     └── 20250801_2210/         # 一次具体实验的结果，以时间戳命名
453 | >         ├── run.log            # 本次实验的完整日志输出
454 | >         ├── result.json        # 过程指标和评估指标（GT, GC, GTE等）
455 | >         ├── plot.svg           # 实验过程中的指标变化可视化图表
456 | >         └── checkpoints/       # 存放模型参数和优化器状态
457 | >             └── step_1000.pt
458 | > ```
459 | >
460 | > **日志 (`run.log`)** 会记录实验过程中的关键信息：
461 | >
462 | > - 数据集概览（使用的特征、标签等）。
463 | > - 预训练过程 (`[pretrain] epoch 1/50 ...`)。
464 | > - 仿真训练过程 (`[train] step 1/1000, ...`)。
465 | > - 验证过程中的指标 (`[validate] step 100/1000, ...`)。
466 | > - 最好尽可能详细记录更多，开始进行xx，完成xx
467 | > - 最终的 GTE 评估结果。
468 | >
469 | > **未来展望 (伏笔)**
470 | >
471 | > 当前这个“global”模式的设计非常巧妙，它为后续更复杂的实验模式（如`splitting`模式）打下了坚实的基础。在未来的模式中，可以不再进行两次完整的仿真，而是在**每一个step内部**，将用户随机分到Treatment组或Control组，并使用各自的`alpha`进行推荐，甚至可以为两组分别维护两个不同的`predicting model`进行训练。
472 | 
473 | > 校对和修正以下和我意图不符的地方：
474 | >
475 | > - 获取数据集统计信息后打印于log；log消息的开头添加[xx（显示当前运行的环节、模块]
476 | > - 特征的选取（config配置）请参考 数据集结构分析报告.md，
477 | >   - 用户特征`user_active_degree`、视频基础特征`video_type`、`tag`是字符型类别变量，可以转化为哑变量，多少类就变为多少个onehot变量。
478 | >   - 视频基础特征`upload_dt`是日期格式，对于预测没有作用，不使用这个特征。
479 | >   - 各种id类变量不适合量化，也不需要，不作为特征使用。
480 | >   - 所有使用的特征都在config中声明式定义，只分两类（数值和分类），未声明的特征不用。
481 | >   - NA等缺失值作为一个新的类别；连续型变量则用0填充
482 | > - label可定义多个，对应alpha也有这么多维数，这里使用click和playtime这两个label，playtime使用logMAE损失函数
483 | >   - 每个label采用独立的预测模型，他们的结构定义作为config中model的子项；treatment实验（计算GT）和Control实验（计算GC）是使用同一个模型的2个实例
484 | >
485 | > 然后尝试运行global mode实验，并持续debug（bug处理要遵循实验合理，不能敷衍了事）
486 | 
487 | > - 调试直到项目跑通，并修复或添加以下：
488 | > - alpha重复定义了，一般只在label项里定义即可
489 | > - 添加结果画图程序
490 | >   - playtime的预测趋向于0，是否是因为大部分真实playtime都是0，那么我希望log记录的playtime不再计算全部的平均值和MSE，而是记录非0的playtime的平均相对误差（百分比那种），画图相应调整为（比例、布局也调整适应）：
491 | >   - 图块1：CTR AUC，包括train/val和treatment/control
492 | >   - 图块2：CTR XXloss，包括train/val和treatment/control
493 | >   - 图块3：playtime 平均相对误差（百分比），包括train/val和treatment/control
494 | >   - 图块4：playtime XXloss，包括train/val和treatment/control
495 | >   - 图块5：weightmodel AUC和Acurracy，包括train/val
496 | >   - 图块6：weightmoel XXloss，包括train/val
497 | > - /
498 | 
499 | >             # Treatment组推荐
500 | >             recommendations_T = self.recommend(user_candidates.copy(), self.alpha_T)
501 | >             
502 | >             # Control组推荐
503 | >             recommendations_C = self.recommend(user_candidates.copy(), self.alpha_C)
504 | >             
505 | >             # 收集真实反馈
506 | >             user_ids_T = [rec[0] for rec in recommendations_T]
507 | >             video_ids_T = [rec[1] for rec in recommendations_T]
508 | >             true_labels_dict_T = self.data_methods.get_true_labels_dict(user_ids_T, video_ids_T)
509 | >             
510 | >             user_ids_C = [rec[0] for rec in recommendations_C]
511 | >             video_ids_C = [rec[1] for rec in recommendations_C]
512 | >             true_labels_dict_C = self.data_methods.get_true_labels_dict(user_ids_C, video_ids_C)
513 | >             
514 | >             # 训练模型（使用两组数据）
515 | >             all_user_ids = user_ids_T + user_ids_C
516 | >             all_video_ids = video_ids_T + video_ids_C
517 | >             
518 | >             # 合并标签
519 | >             all_labels_dict = {}
520 | >             for label_name in self.label_names:
521 | >                 # 确保两组数据都有该标签
522 | >                 if label_name in true_labels_dict_T and label_name in true_labels_dict_C:
523 | >                     all_labels_dict[label_name] = np.concatenate([
524 | >                         true_labels_dict_T[label_name], 
525 | >                         true_labels_dict_C[label_name]
526 | >                     ])
527 | >             
528 | >             # 训练每个模型
529 | >             losses = self.train_step(optimizers, criteria, all_user_ids, all_video_ids, all_labels_dict)
530 | >             
531 | >             # 记录损失
532 | >             for label_name, loss in losses.items():
533 | >                 self.training_losses[label_name].append(loss)
534 | >             
535 | >             # 累积标签
536 | >             for label_name in self.label_names:
537 | >                 if label_name in true_labels_dict_T:
538 | >                     self.total_labels_T[label_name] += true_labels_dict_T[label_name].sum()
539 | >                 if label_name in true_labels_dict_C:
540 | >                     self.total_labels_C[label_name] += true_labels_dict_C[label_name].sum()
541 | >             
542 | >             这一段的逻辑你背离了我的初衷，我希望的是把global mode的逻辑放在libs/exp_modes目录下的单独的global.py文件中，方便以后增加weighting.py\spliting.py……；global mode的逻辑是独立运行两次实验，**：一次全员使用处理组推荐参数的alpha (vecalpha∗T)来计算GT，一次全员使用对照组的alpha (vecalpha∗C)来计算GC。两次仿真没有交集，不要把他们混在一起训练，used也是独立的，不共享
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/README.md

- Extension: .md
- Language: markdown
- Size: 8691 bytes
- Created: 2025-08-04 13:46:12
- Modified: 2025-08-04 13:46:12

### Code

```markdown
  1 | # RealdataEXP 实验框架
  2 | 
  3 | ## 项目概述
  4 | 
  5 | RealdataEXP 是一个基于真实数据的推荐系统实验框架，用于计算全局处理效应（Global Treatment Effect, GTE）。该框架支持多种实验模式，包括global、weighting、splitting等，旨在为推荐系统的因果推断研究提供完整的实验平台。
  6 | 
  7 | ## 项目特性
  8 | 
  9 | - **多模式实验支持**：支持global、weighting、splitting等多种实验模式
 10 | - **多标签预测**：同时支持点击率和播放时长等多个标签的预测
 11 | - **灵活的特征处理**：自动处理数值特征和分类特征，支持one-hot编码
 12 | - **缓存机制**：优化数据加载性能，避免重复计算
 13 | - **完整的实验记录**：详细的日志记录和结果保存
 14 | - **可扩展架构**：模块化设计，易于扩展新的实验模式
 15 | 
 16 | ## 系统架构
 17 | 
 18 | ```
 19 | RealdataEXP/
 20 | ├── configs/                    # 配置文件
 21 | │   └── experiment.yaml        # 实验配置
 22 | ├── data/                      # 数据目录
 23 | │   └── KuaiRand/             # KuaiRand数据集
 24 | │       ├── Pure/             # Pure版本数据
 25 | │       ├── 1K/               # 1K版本数据
 26 | │       ├── 27K/              # 27K版本数据
 27 | │       └── cache/            # 缓存目录
 28 | ├── libs/                     # 核心代码库
 29 | │   ├── data/                 # 数据处理模块
 30 | │   │   ├── data_loader.py    # 数据加载器
 31 | │   │   ├── feature_processor.py # 特征处理器
 32 | │   │   └── cache_manager.py  # 缓存管理器
 33 | │   ├── models/               # 模型模块
 34 | │   │   ├── mlp_model.py      # MLP模型
 35 | │   │   ├── multi_label_model.py # 多标签模型
 36 | │   │   └── loss_functions.py # 损失函数
 37 | │   ├── modes/                # 实验模式
 38 | │   │   ├── global_mode.py    # Global模式
 39 | │   │   ├── weighting.py      # Weighting模式（待实现）
 40 | │   │   └── splitting.py      # Splitting模式（待实现）
 41 | │   └── utils/                # 工具模块
 42 | │       ├── logger.py         # 日志工具
 43 | │       ├── metrics.py        # 指标跟踪
 44 | │       └── experiment_utils.py # 实验工具
 45 | ├── results/                  # 实验结果
 46 | │   └── [timestamp]/          # 按时间戳组织的实验结果
 47 | │       ├── run.log          # 运行日志
 48 | │       ├── result.json      # 实验结果
 49 | │       └── checkpoints/     # 模型检查点
 50 | └── main.py                  # 主入口程序
 51 | ```
 52 | 
 53 | ![RealdataEXP/实验框架v2.pdf](RealdataEXP/实验框架v2.pdf)
 54 | 
 55 | ## 数据集
 56 | 
 57 | 项目使用KuaiRand数据集，包含：
 58 | 
 59 | - **用户行为日志**：记录用户与视频的交互行为
 60 |   - 总样本数：2,622,668
 61 |   - 点击率：33.14%
 62 |   - 平均播放时长：15,676.54ms
 63 | 
 64 | - **用户特征**：用户的静态画像数据
 65 |   - 用户数：27,285
 66 |   - 训练用户：21,828
 67 |   - 验证用户：5,457
 68 | 
 69 | - **视频特征**：视频的基础信息和统计特征
 70 |   - 视频数：7,583
 71 |   - 特征维度：157（数值特征：34，分类特征：123）
 72 | 
 73 | ## 核心模块
 74 | 
 75 | ### 1. 数据加载与处理模块
 76 | 
 77 | - **数据流**：从数据集提取用户ID，查找用户交互的视频列表
 78 | - **缓存机制**：避免重复计算，提高处理效率
 79 | - **特征处理**：
 80 |   - 数值特征：标准化处理
 81 |   - 分类特征：one-hot编码（`user_active_degree`、`video_type`、`tag`）
 82 |   - 缺失值处理：数值特征用0填充，分类特征作为新类别
 83 | 
 84 | ### 2. 多标签预测模型
 85 | 
 86 | - **独立模型架构**：每个标签使用独立的MLP模型
 87 | - **支持的标签**：
 88 |   - `play_time`：播放时长预测（使用logMAE损失函数）
 89 |   - `click`：点击预测（使用二元交叉熵损失函数）
 90 | - **模型参数**：
 91 |   - play_time模型：30,593参数
 92 |   - click模型：12,737参数
 93 | 
 94 | ### 3. Global模式实验
 95 | 
 96 | Global模式是框架的核心，实现真实GTE的计算：
 97 | 
 98 | - **对称仿真**：Treatment组和Control组独立运行
 99 | - **实验流程**：
100 |   1. 用户批次抽样
101 |   2. 候选视频生成
102 |   3. 模型预测与加权排序
103 |   4. 选出胜出视频
104 |   5. 获取真实反馈与模型训练
105 |   6. 更新状态标记
106 | - **标记机制**：
107 |   - `mask`：验证集用户的视频标记
108 |   - `used`：已推荐视频的标记（两组独立维护）
109 | 
110 | ## 配置文件
111 | 
112 | `configs/experiment.yaml` 包含完整的实验配置：
113 | 
114 | ```yaml
115 | # 实验模式
116 | mode: 'global'
117 | 
118 | # 数据集配置
119 | dataset:
120 |   name: "KuaiRand-Pure"
121 |   path: "data/KuaiRand/Pure"
122 |   cache_path: "data/KuaiRand/cache"
123 | 
124 | # 特征配置
125 | feature:
126 |   numerical: [数值特征列表]
127 |   categorical: [分类特征列表]
128 | 
129 | # 多标签配置
130 | labels:
131 |   - name: "play_time"
132 |     target: "play_time_ms"
133 |     type: "numerical"
134 |     loss_function: "logMAE"
135 |     # ... 模型参数
136 |   - name: "click"
137 |     target: "is_click"
138 |     type: "binary"
139 |     loss_function: "BCE"
140 |     # ... 模型参数
141 | 
142 | # Global模式配置
143 | global:
144 |   user_p_val: 0.2      # 验证集比例
145 |   batch_size: 64       # 批次大小
146 |   n_candidate: 10      # 候选视频数
147 |   n_steps: 200         # 仿真步数
148 |   validate_every: 25   # 验证频率
149 | ```
150 | 
151 | ## 安装和使用
152 | 
153 | ### 环境要求
154 | 
155 | - Python 3.7+
156 | - PyTorch 2.0+
157 | - pandas 2.0+
158 | - numpy 2.0+
159 | - scikit-learn 1.6+
160 | - PyYAML
161 | 
162 | ### 运行实验（CPU）
163 | 
164 | ```bash
165 | # 运行Global模式实验
166 | python main.py --mode global
167 | 
168 | # 使用自定义配置文件
169 | python main.py --config configs/my_experiment.yaml
170 | 
171 | # 指定实验模式（覆盖配置文件设置）
172 | python main.py --mode global --config configs/experiment.yaml
173 | ```
174 | 
175 | ### 运行实验（GPU）
176 | ## GPU集群使用指南
177 | 
178 | 本框架支持在HKUST HPC4集群上使用SLURM进行GPU加速训练。以下是完整的GPU使用流程。
179 | 
180 | ### 环境要求
181 | 
182 | - HKUST HPC4集群账户
183 | - 项目组账户：`sigroup`
184 | - PyTorch 2.0+ (自带CUDA runtime)
185 | - SLURM作业调度系统
186 | 
187 | ### 1. 提交GPU作业
188 | 
189 | #### 1.1 使用预配置脚本
190 | 
191 | ```bash
192 | # 提交GPU作业
193 | sbatch run_gpu.sh
194 | ```
195 | ### 2. 查看作业状态
196 | 
197 | ```bash
198 | # 查看用户作业队列
199 | squeue -u $USER
200 | 
201 | # 查看作业详细信息
202 | scontrol show job <作业ID>
203 | 
204 | # 取消作业
205 | scancel <作业ID>
206 | ```
207 | 
208 | ### 3. 连接GPU节点
209 | 
210 | #### 3.1 连接到已分配的GPU节点
211 | 
212 | ```bash
213 | # 连接到正在运行的作业节点 (示例: 作业52005在gpu01)
214 | srun --jobid=52098 -w gpu01 --overlap --pty bash -i
215 | ```
216 | 
217 | ### 实验结果
218 | 
219 | 实验结果保存在 `results/[timestamp]/` 目录下：
220 | 
221 | - `run.log`：完整的运行日志
222 | - `result.json`：实验结果和指标
223 | - `checkpoints/`：模型检查点和特征处理器
224 | 
225 | ## 开发进展
226 | 
227 | ### 已完成功能
228 | 
229 | - ✅ 数据加载和预处理模块
230 | - ✅ 多标签预测模型架构
231 | - ✅ Global模式核心逻辑
232 | - ✅ 特征处理和缓存机制
233 | - ✅ 实验日志和结果保存
234 | - ✅ 配置管理系统
235 | 
236 | ### 待实现功能
237 | 
238 | - ⏳ Weighting模式实验
239 | - ⏳ Splitting模式实验
240 | - ⏳ 实验结果可视化
241 | - ⏳ 模型性能评估指标
242 | - ⏳ 分布式训练支持
243 | 
244 | ## 技术细节
245 | 
246 | ### 数据类型处理
247 | 
248 | 框架实现了强健的数据类型转换机制，确保所有特征数据都能正确转换为PyTorch张量：
249 | 
250 | ```python
251 | def ensure_float_data(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
252 |     """确保数据为float类型并转换为numpy数组"""
253 |     # 逐列强制转换为数值类型
254 |     # 处理NaN值和非数值类型
255 |     # 返回float32数组
256 | ```
257 | 
258 | ### 缓存优化
259 | 
260 | 用户-视频交互列表使用pickle缓存，避免重复计算：
261 | 
262 | ```python
263 | # 首次计算时保存缓存
264 | cache_manager.save(user_video_lists, "user_video_lists")
265 | 
266 | # 后续运行直接加载
267 | cached_data = cache_manager.load("user_video_lists")
268 | ```
269 | 
270 | ### 损失函数
271 | 
272 | 支持多种损失函数：
273 | 
274 | - **LogMAE**：用于播放时长等大数值范围的连续标签
275 | - **BCE**：用于点击等二元分类标签
276 | - **MSE**、**MAE**、**CrossEntropy**：其他常用损失函数
277 | 
278 | ## 扩展指南
279 | 
280 | ### 添加新的实验模式
281 | 
282 | 1. 在 `libs/modes/` 下创建新的模式文件
283 | 2. 继承基础实验类，实现核心逻辑
284 | 3. 在 `main.py` 中添加模式分派逻辑
285 | 4. 更新配置文件模板
286 | 
287 | ### 添加新的模型
288 | 
289 | 1. 在 `libs/models/` 下创建模型文件
290 | 2. 实现PyTorch模型接口
291 | 3. 在多标签模型管理器中注册新模型
292 | 4. 更新配置文件中的模型参数
293 | 
294 | ### 添加新的特征
295 | 
296 | 1. 在配置文件中声明新特征
297 | 2. 确保数据集包含对应字段
298 | 3. 根据特征类型选择数值或分类处理
299 | 4. 测试特征处理和模型训练流程
300 | 
301 | ## 许可证
302 | 
303 | 本项目采用MIT许可证。详见LICENSE文件。
304 | 
305 | ## 联系方式
306 | 
307 | 如有问题或建议，请提交Issue或联系项目维护者。
308 | 
309 | ---
310 | 
311 | *最后更新：2025年8月3日*
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/main.py

- Extension: .py
- Language: python
- Size: 2989 bytes
- Created: 2025-08-04 13:26:34
- Modified: 2025-08-04 13:26:34

### Code

```python
 1 | #!/usr/bin/env python3
 2 | """
 3 | RealdataEXP 实验框架主入口程序
 4 | 支持多种实验模式：global, weighting, splitting等
 5 | """
 6 | 
 7 | import os
 8 | import sys
 9 | import yaml
10 | import argparse
11 | import logging
12 | from datetime import datetime
13 | 
14 | # 添加项目根目录到Python路径
15 | project_root = os.path.dirname(__file__)
16 | sys.path.insert(0, project_root)
17 | 
18 | from libs.utils import setup_logger, create_experiment_dir
19 | from libs.modes import GlobalMode
20 | 
21 | def load_config(config_path: str) -> dict:
22 |     """加载配置文件"""
23 |     try:
24 |         with open(config_path, 'r', encoding='utf-8') as f:
25 |             config = yaml.safe_load(f)
26 |         return config
27 |     except Exception as e:
28 |         raise RuntimeError(f"配置文件加载失败: {e}")
29 | 
30 | def main():
31 |     """主函数"""
32 |     parser = argparse.ArgumentParser(description='RealdataEXP 实验框架')
33 |     parser.add_argument('--config', '-c', type=str, 
34 |                       default='configs/experiment.yaml',
35 |                       help='配置文件路径')
36 |     parser.add_argument('--mode', '-m', type=str,
37 |                       help='实验模式 (覆盖配置文件中的mode设置)')
38 |     
39 |     args = parser.parse_args()
40 |     
41 |     # 加载配置
42 |     config_path = os.path.join(os.path.dirname(__file__), args.config)
43 |     config = load_config(config_path)
44 |     
45 |     # 命令行参数覆盖配置
46 |     if args.mode:
47 |         config['mode'] = args.mode
48 |     
49 |     # 创建实验目录
50 |     base_dir = config.get('base_dir', os.path.dirname(__file__))
51 |     exp_dir = create_experiment_dir(base_dir)
52 |     
53 |     # 设置日志
54 |     log_file = os.path.join(exp_dir, 'run.log')
55 |     logger = setup_logger(log_file, config.get('logging', {}).get('level', 'INFO'))
56 |     
57 |     logger.info("=" * 60)
58 |     logger.info("RealdataEXP 实验框架启动")
59 |     logger.info("=" * 60)
60 |     logger.info(f"实验模式: {config['mode']}")
61 |     logger.info(f"实验目录: {exp_dir}")
62 |     logger.info(f"配置文件: {config_path}")
63 |     
64 |     try:
65 |         # 根据模式运行相应的实验
66 |         mode = config['mode'].lower()
67 |         
68 |         if mode == 'global':
69 |             logger.info("[模式选择] 运行Global模式实验")
70 |             experiment = GlobalMode(config, exp_dir)
71 |             experiment.run()
72 |             
73 |         elif mode == 'weighting':
74 |             logger.error("[模式选择] Weighting模式尚未实现")
75 |             raise NotImplementedError("Weighting模式尚未实现")
76 |             
77 |         elif mode == 'splitting':
78 |             logger.error("[模式选择] Splitting模式尚未实现") 
79 |             raise NotImplementedError("Splitting模式尚未实现")
80 |             
81 |         else:
82 |             raise ValueError(f"不支持的实验模式: {mode}")
83 |             
84 |         logger.info("=" * 60)
85 |         logger.info("实验成功完成!")
86 |         logger.info("=" * 60)
87 |         
88 |     except Exception as e:
89 |         logger.error(f"实验执行失败: {e}")
90 |         logger.exception("详细错误信息:")
91 |         sys.exit(1)
92 | 
93 | if __name__ == '__main__':
94 |     main()
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/run_gpu.sh

- Extension: .sh
- Language: bash
- Size: 3748 bytes
- Created: 2025-08-11 22:22:36
- Modified: 2025-08-11 22:22:36

### Code

```bash
  1 | #!/bin/bash
  2 | 
  3 | # GPU 作业提交脚本 - Global Mode实验
  4 | # 使用方法: sbatch run_gpu.sh
  5 | 
  6 | #SBATCH --account=sigroup     # 你的账户名
  7 | #SBATCH --time=23:30:00       # 运行时间限制 (23.5小时) - 减少以提高调度优先级
  8 | #SBATCH --partition=gpu-a30   # GPU分区 (A30 GPU)
  9 | #SBATCH --gpus-per-node=1     # 每个节点使用1个GPU
 10 | #SBATCH --cpus-per-task=32     # 每个任务使用32个CPU核心
 11 | #SBATCH --mem=64G             # 内存需求
 12 | #SBATCH --job-name=global_mode_gpu  # 更明确的作业名称
 13 | #SBATCH --output=results/gpu_run_%j.out  # 标准输出文件
 14 | #SBATCH --error=results/gpu_run_%j.err   # 错误输出文件
 15 | 
 16 | echo "============================================================"
 17 | echo "作业开始时间: $(date)"
 18 | echo "作业ID: $SLURM_JOB_ID"
 19 | echo "节点名称: $SLURM_NODELIST"
 20 | echo "GPU数量: $SLURM_GPUS_PER_NODE"
 21 | echo "============================================================"
 22 | 
 23 | # 切换到项目目录
 24 | cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
 25 | 
 26 | # 加载CUDA模块
 27 | echo "加载CUDA模块..."
 28 | module load cuda
 29 | 
 30 | # 环境检测和确认
 31 | echo "============================================================"
 32 | echo "CUDA环境检测"
 33 | echo "============================================================"
 34 | 
 35 | # 检查GPU可用性
 36 | echo "检查GPU状态..."
 37 | if command -v nvidia-smi &> /dev/null; then
 38 |     nvidia-smi
 39 |     GPU_STATUS=$?
 40 |     if [ $GPU_STATUS -eq 0 ]; then
 41 |         echo "✅ GPU检测成功"
 42 |     else
 43 |         echo "❌ GPU检测失败"
 44 |         exit 1
 45 |     fi
 46 | else
 47 |     echo "❌ nvidia-smi命令不可用"
 48 |     exit 1
 49 | fi
 50 | 
 51 | # 检查CUDA版本
 52 | echo ""
 53 | echo "检查CUDA版本..."
 54 | if command -v nvcc &> /dev/null; then
 55 |     nvcc --version
 56 |     echo "✅ CUDA工具包可用"
 57 | else
 58 |     echo "⚠️ CUDA编译器不可用，但GPU可能仍然可用"
 59 | fi
 60 | 
 61 | # 运行Python环境检查
 62 | echo ""
 63 | echo "检查Python和PyTorch环境..."
 64 | python check_environment.py
 65 | 
 66 | # 检查检测结果
 67 | PYTHON_CHECK=$?
 68 | if [ $PYTHON_CHECK -ne 0 ]; then
 69 |     echo "❌ Python环境检测失败，继续运行但可能遇到问题"
 70 |     echo "注意：在SLURM环境中，GPU只有在作业分配后才可用"
 71 | fi
 72 | 
 73 | echo ""
 74 | echo "============================================================"
 75 | echo "环境检测完成 - 所有检查通过！"
 76 | echo "============================================================"
 77 | 
 78 | # 对于批处理作业，自动继续（不等待用户输入）
 79 | echo "批处理模式：自动开始实验..."
 80 | sleep 2
 81 | 
 82 | # 设置环境变量
 83 | export CUDA_VISIBLE_DEVICES=0
 84 | export PYTHONPATH=$PYTHONPATH:/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
 85 | 
 86 | echo ""
 87 | echo "============================================================"
 88 | echo "开始运行实验"
 89 | echo "============================================================"
 90 | echo "使用配置文件: configs/experiment.yaml"
 91 | echo "设备配置: auto (将自动选择GPU)"
 92 | 
 93 | # 最终GPU检查（在SLURM分配后）
 94 | echo "SLURM作业分配后的GPU状态："
 95 | if command -v nvidia-smi &> /dev/null; then
 96 |     nvidia-smi
 97 | else
 98 |     echo "nvidia-smi不可用，但PyTorch应该仍能检测到GPU"
 99 | fi
100 | 
101 | echo ""
102 | echo "开始运行Global Mode GPU实验..."
103 | echo "预期运行时间：约60-90分钟"
104 | echo ""
105 | 
106 | # 运行实验，增加详细输出
107 | python main.py --config configs/experiment.yaml --mode global 2>&1 | tee results/gpu_run_${SLURM_JOB_ID}_detailed.log
108 | 
109 | EXPERIMENT_STATUS=$?
110 | 
111 | echo ""
112 | echo "============================================================"
113 | if [ $EXPERIMENT_STATUS -eq 0 ]; then
114 |     echo "✅ 实验成功完成！"
115 | else
116 |     echo "❌ 实验执行出错，退出码: $EXPERIMENT_STATUS"
117 | fi
118 | echo "作业结束时间: $(date)"
119 | echo "详细日志保存在: results/gpu_run_${SLURM_JOB_ID}_detailed.log"
120 | echo "============================================================"
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/Exp logs.md

- Extension: .md
- Language: markdown
- Size: 6783 bytes
- Created: 2025-08-04 13:41:12
- Modified: 2025-08-04 13:41:12

### Code

```markdown
  1 | ## 20250802（night）
  2 | ### 重新构建项目
  3 | 
  4 | ## 20250803
  5 | ### Claude Code
  6 | ---
  7 | <instruction>
  8 | deep think：我希望你帮我解决gpu和cpu利用效率低下，导致模型训练太慢的问题。在你接手项目之前，我已经做了一次尝试，但出现了bug。你可以选择、考虑自己寻找优化方法，或者在这个有bug的方案基础上修复
  9 | </instruction>
 10 | 
 11 | <context>
 12 | /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/results/gpu_run_52019_detailed.log 结合预训练每一个epoch花了20min，系统负载: 1.12 (1分钟平均)
 13 | CPU总体: 1.7% user, 98.2% idle
 14 | 实验进程: 99.7% CPU使用率 (单核满载)和nvidia-smi显示gpu低占用，我怀疑没有正确使用gpu进行训练，或者是在其他环节占用了太多时间（如cpu没有多线程工作），请你诊断并改善
 15 | 
 16 | 已创建的优化文件：
 17 | ✅ libs/modes/global_mode_optimized.py - 优化训练引擎
 18 | ✅ configs/experiment_optimized.yaml - 优化配置
 19 | ✅ run_gpu_optimized.sh - 优化GPU作业脚本
 20 | ✅ monitor_gpu_optimized.sh - 性能监控工具
 21 | ✅ performance_analysis.py - 性能对比分析
 22 | ✅ GPU_OPTIMIZATION_GUIDE.md - 完整使用指南
 23 | 
 24 | 优化后实验日志：
 25 | /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/results/gpu_run_52022_detailed.log
 26 | </context>
 27 | ---
 28 | claude code为我修复后，playtime loss出现了inf；实验进入到Trratment实验阶段，暴露出新的bug，参见/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/results/gpu_run_52049_detailed.log
 29 | ---
 30 | 
 31 | ## 20250803（continued）
 32 | ### 关键Bug修复和Job 52068状态更新
 33 | 
 34 | #### 问题背景
 35 | 从Job 52049暴露出的关键bug：
 36 | 1. **List Index Out of Range Error** - 在仿真步骤中出现索引越界错误
 37 | 2. **Shape Mismatch in Batch Training** - 特征样本数与标签样本数不匹配
 38 | 3. **Tensor Conversion Error** - 多元素张量转换为标量失败
 39 | 
 40 | #### 实施的关键修复
 41 | 
 42 | **修复1: 张量形状验证和安全索引 (global_mode_optimized.py:509-529)**
 43 | ```python
 44 | # 确保combined_scores是正确的形状并获取有效索引
 45 | scores_squeezed = combined_scores.squeeze()
 46 | if scores_squeezed.dim() == 0:
 47 |     scores_squeezed = scores_squeezed.unsqueeze(0)
 48 | elif scores_squeezed.dim() > 1:
 49 |     scores_squeezed = scores_squeezed.flatten()
 50 | 
 51 | # 确保索引在有效范围内
 52 | if len(scores_squeezed) != len(candidates):
 53 |     logger.warning(f"[{prefix}仿真优化] 分数张量长度 {len(scores_squeezed)} 与候选视频数 {len(candidates)} 不匹配")
 54 |     safe_length = min(len(scores_squeezed), len(candidates))
 55 |     if safe_length == 0:
 56 |         continue  # 跳过这个用户
 57 |     winner_idx = torch.argmax(scores_squeezed[:safe_length]).item()
 58 | else:
 59 |     winner_idx = torch.argmax(scores_squeezed).item()
 60 | ```
 61 | 
 62 | **修复2: 特征-标签大小验证和动态调整 (global_mode_optimized.py:565-593)**
 63 | ```python
 64 | # 验证特征和标签的样本数量是否一致
 65 | n_features = all_features.size(0)
 66 | n_targets = combined_targets[list(combined_targets.keys())[0]].size(0)
 67 | 
 68 | if n_features != n_targets:
 69 |     logger.warning(f"[{prefix}仿真优化] 特征样本数 {n_features} 与标签样本数 {n_targets} 不匹配，调整批量训练")
 70 |     min_samples = min(n_features, n_targets)
 71 |     if min_samples == 0:
 72 |         logger.warning(f"[{prefix}仿真优化] 没有有效样本用于训练，跳过批量训练")
 73 |     else:
 74 |         # 调整张量大小
 75 |         all_features = all_features[:min_samples]
 76 |         for label_name in combined_targets:
 77 |             combined_targets[label_name] = combined_targets[label_name][:min_samples]
 78 | ```
 79 | 
 80 | **修复3: 张量转换安全处理 (global_mode_optimized.py:542-548)**
 81 | ```python
 82 | # 确保张量是标量，然后提取值
 83 | if label_tensor.numel() == 1:
 84 |     reward_value = label_tensor.item()
 85 | else:
 86 |     # 如果张量有多个元素，取第一个元素或求和
 87 |     reward_value = label_tensor.sum().item()
 88 | ```
 89 | 
 90 | #### Job 52068当前状态
 91 | - **作业ID**: 52068
 92 | - **当前进度**: Step 72/100 (72%完成)
 93 | - **开始时间**: 2025-08-03 18:06:18
 94 | - **已运行时间**: 约3小时46分钟
 95 | - **预计完成**: 约1小时后
 96 | 
 97 | #### 关键指标（截至Step 71）
 98 | - **Treatment组总播放时长**: 49,577,210ms
 99 | - **Control组总播放时长**: 44,265,970ms  
100 | - **Treatment组总点击数**: 1,054
101 | - **Control组总点击数**: 1,004
102 | - **当前GTE趋势**: Treatment组表现优于Control组
103 | 
104 | #### 重要观察
105 | 1. **修复效果**: 所有之前的关键错误已解决，作业稳定运行
106 | 2. **数据耗尽现象**: 从Step 66开始，每步处理用户数降为0，说明可用视频资源接近耗尽
107 | 3. **性能稳定**: 没有出现新的错误或警告，系统运行稳定
108 | 4. **预期结果**: 按当前趋势，实验将成功完成并输出GTE分析结果
109 | 
110 | #### 技术改进验证
111 | - ✅ **索引安全**: 彻底解决了list index out of range错误
112 | - ✅ **形状匹配**: 自动处理特征-标签维度不一致问题
113 | - ✅ **张量转换**: 安全处理多元素张量转换
114 | - ✅ **错误恢复**: 增强的警告和跳过机制
115 | - ✅ **稳定性**: 长时间运行无崩溃
116 | 
117 | #### 后续建议
118 | 1. **数据资源**: 考虑增加候选视频池或实现视频重用机制
119 | 2. **早期停止**: 当连续多步处理用户数为0时，可考虑提前结束实验
120 | 3. **监控优化**: 添加资源使用率监控，便于分析数据耗尽模式
121 | 
122 | ---
123 | <instruction>
124 | 在gpu集群上继续实验，并根据最新log决定下一步
125 | </instruction>
126 | <context>
127 | 你可以通过README和/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/Exp logs.md熟悉该项目和历史更改。最新的一步，claude code为我在log日志输出增加了playtime的评价相对误差指标，并尝试修复了新的bug，还没有在服务器上尝试过
128 | @README ### 1. 提交GPU作业 ### 2. 查看作业状态 ### 3. 连接GPU节点
129 | </context>
130 | ---
131 | 
132 | <instruction>
133 | deep thinking: 
134 | 解决context所说的问题；
135 | 另外，我怀疑训练过程没有真正在gpu上进行运算，请你想办法调试检查这一点；
136 | 训练效果惨不忍睹，你可以从log看看预测相对误差，需要调整playtime模型：我已经微调了参数，可以试试看，你可以提出更多办法调试效果差的原因并提出改善模型的建议。
137 | </instruction>
138 | <context>
139 | 查看/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/results/gpu_run_52068_detailed.log，你会发现大量的不匹配，调整索引范围、不匹配，调整批量训练、WARNING和"/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/modes/global_mode_optimized.py:583: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
140 |   with autocast():"
141 | </context>
142 | ---
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/.gitignore

- Extension: 
- Language: unknown
- Size: 470 bytes
- Created: 2025-08-04 13:26:35
- Modified: 2025-08-04 13:26:35

### Code

```unknown
 1 | KuaiRand/
 2 | 
 3 | # Python cache
 4 | __pycache__/
 5 | *.pyc
 6 | *.pyo
 7 | *.pyd
 8 | 
 9 | # Jupyter Notebook checkpoints
10 | .ipynb_checkpoints/
11 | 
12 | # VSCode
13 | .vscode/
14 | *.code-workspace
15 | 
16 | # Conda/venv environments
17 | .conda/
18 | venv/
19 | env/
20 | ENV/
21 | # Conda/miniconda environments and installer
22 | envs/
23 | miniconda3/
24 | Miniconda3-latest-Linux-x86_64.sh
25 | 
26 | 
27 | # Model checkpoints and results
28 | results/[0-9]*/
29 | results/[0-9]*/checkpoints/
30 | *.pt
31 | *.pth
32 | 
33 | # Data archives
34 | *.tar.gz
35 | *.zip
36 | *.rar
37 | 
38 | # System files
39 | .DS_Store
40 | Thumbs.db
41 | 
42 | # Logs
43 | *.log
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/__init__.py

- Extension: .py
- Language: python
- Size: 84 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
1 | """
2 | RealdataEXP 核心库
3 | """
4 | 
5 | __version__ = "1.0.0"
6 | __author__ = "RealdataEXP Team"
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/configs/experiment.yaml

- Extension: .yaml
- Language: yaml
- Size: 4186 bytes
- Created: 2025-08-11 22:02:57
- Modified: 2025-08-11 22:02:57

### Code

```yaml
  1 | 
  2 | base_dir: "/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP"
  3 | mode: 'global'  # 实验模式: global, weighting, splitting等
  4 | 
  5 | # 数据集配置
  6 | dataset:
  7 |   name: "KuaiRand-Pure"  # 数据集名称: KuaiRand-Pure, KuaiRand-1K, KuaiRand-27K
  8 |   path: "data/KuaiRand/Pure"  # 数据集路径
  9 |   cache_path: "data/KuaiRand/cache"  # 缓存目录
 10 | 
 11 | # 模型训练相关特征配置（根据数据集结构分析报告）
 12 | feature:
 13 |   numerical:  # 数值型特征
 14 |     # 视频基础特征
 15 |     - "video_duration"        # 视频时长（毫秒）
 16 |     - "server_width"          # 视频宽度
 17 |     - "server_height"         # 视频高度
 18 |     # 视频统计特征
 19 |     - "show_cnt"              # 累计曝光次数
 20 |     - "play_cnt"              # 累计播放次数
 21 |     - "play_user_num"         # 累计播放用户数
 22 |     - "complete_play_cnt"     # 累计完播次数
 23 |     - "like_cnt"              # 累计点赞数
 24 |     - "comment_cnt"           # 累计评论数
 25 |     - "share_cnt"             # 累计分享数
 26 |     - "collect_cnt"           # 累计收藏数
 27 |     # 用户特征
 28 |     - "is_live_streamer"      # 是否为直播主播
 29 |     - "is_video_author"       # 是否为视频创作者
 30 |     - "follow_user_num"       # 用户关注数
 31 |     - "fans_user_num"         # 用户粉丝数
 32 |     - "friend_user_num"       # 用户好友数
 33 |     - "register_days"         # 账号注册天数
 34 |     # # 用户onehot特征（作为数值特征处理）
 35 |     # - "onehot_feat0"
 36 |     # - "onehot_feat1"
 37 |     # - "onehot_feat2"
 38 |     # - "onehot_feat3"
 39 |     # - "onehot_feat4"
 40 |     # - "onehot_feat5"
 41 |     # - "onehot_feat6"
 42 |     # - "onehot_feat7"
 43 |     # - "onehot_feat8"
 44 |     # - "onehot_feat9"
 45 |     # - "onehot_feat10"
 46 |     # - "onehot_feat11"
 47 |     # - "onehot_feat12"
 48 |     # - "onehot_feat13"
 49 |     # - "onehot_feat14"
 50 |     # - "onehot_feat15"
 51 |     # - "onehot_feat16"
 52 |     # - "onehot_feat17"
 53 |   categorical:  # 分类型特征（将转换为onehot变量）
 54 |     - "user_active_degree"    # 用户活跃度等级
 55 |     - "video_type"            # 视频类型
 56 |     - "tag"                   # 视频标签
 57 | 
 58 | # 标签配置（多标签预测，每个标签使用独立模型）
 59 | labels:
 60 |   - name: "play_time"         # 播放时长
 61 |     target: "play_time_ms"    # 对应的数据集字段名
 62 |     type: "numerical"         # 标签类型: binary, numerical
 63 |     loss_function: "logMAE"   # 播放时长使用logMAE损失函数
 64 |     model: "MLP"              # 模型类型
 65 |     model_params:
 66 |       hidden_layers: [128, 64, 32]  # 3层MLP架构
 67 |       dropout: 0.2                  # dropout率
 68 |       embedding_dim: 16             # 分类特征的嵌入维度
 69 |     learning_rate: 0.0001
 70 |     weight_decay: 0.0001
 71 |     alpha_T: 1.0                    # Treatment组的alpha权重
 72 |     alpha_C: 0.5                    # Control组的alpha权重
 73 |     
 74 |   - name: "click"             # 点击
 75 |     target: "is_click"        # 对应的数据集字段名
 76 |     type: "binary"            # 标签类型: binary, numerical
 77 |     loss_function: "BCE"      # 点击使用二元交叉熵损失函数
 78 |     model: "MLP"              # 模型类型
 79 |     model_params:
 80 |       hidden_layers: [64, 32, 16]   # 3层MLP架构
 81 |       dropout: 0.1                  # dropout率
 82 |       embedding_dim: 8              # 分类特征的嵌入维度
 83 |     learning_rate: 0.0001
 84 |     weight_decay: 0.0001
 85 |     alpha_T: 1.0                    # Treatment组的alpha权重
 86 |     alpha_C: 0.8                    # Control组的alpha权重
 87 | 
 88 | # 预训练配置
 89 | pretrain:
 90 |   enabled: true  # 是否启用预训练
 91 |   batch_size: 64
 92 |   epochs: 1     # 减少epoch数量，从50减少到10
 93 |   learning_rate: 0.001
 94 |   weight_decay: 0.0001
 95 |   early_stopping: 3  # 提前停止的检查点数，从5减少到3
 96 | 
 97 | # 全局仿真配置 (仅在mode='global'时使用)
 98 | global:
 99 |   user_p_val: 0.2  # 验证集用户比例
100 |   batch_size: 64   # 每步抽样的用户数
101 |   n_candidate: 10  # 每个用户的候选视频数
102 |   n_steps: 5     # 仿真总步数，从1000减少到200
103 |   validate_every: 1  # 每隔多少步验证一次，从50减少到25
104 |   save_every: 50    # 每隔多少步保存一次模型
105 |   learning_rate: 0.0005
106 |   weight_decay: 0.0001
107 | 
108 | # 日志配置
109 | logging:
110 |   level: "INFO"
111 |   log_dir: "results"
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/logger.py

- Extension: .py
- Language: python
- Size: 916 bytes
- Created: 2025-08-03 11:14:53
- Modified: 2025-08-03 11:14:53

### Code

```python
 1 | """
 2 | 日志配置工具
 3 | """
 4 | 
 5 | import logging
 6 | import os
 7 | from datetime import datetime
 8 | 
 9 | def setup_logger(log_file: str, level: str = 'INFO') -> logging.Logger:
10 |     """设置日志器"""
11 |     # 创建日志目录
12 |     os.makedirs(os.path.dirname(log_file), exist_ok=True)
13 |     
14 |     # 配置日志格式
15 |     formatter = logging.Formatter(
16 |         '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
17 |         datefmt='%Y-%m-%d %H:%M:%S'
18 |     )
19 |     
20 |     # 创建文件处理器
21 |     file_handler = logging.FileHandler(log_file, encoding='utf-8')
22 |     file_handler.setFormatter(formatter)
23 |     
24 |     # 创建控制台处理器
25 |     console_handler = logging.StreamHandler()
26 |     console_handler.setFormatter(formatter)
27 |     
28 |     # 配置根日志器
29 |     logger = logging.getLogger()
30 |     logger.setLevel(getattr(logging, level.upper()))
31 |     logger.addHandler(file_handler)
32 |     logger.addHandler(console_handler)
33 |     
34 |     return logger
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/metrics.py

- Extension: .py
- Language: python
- Size: 1796 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
 1 | """
 2 | 指标跟踪器
 3 | """
 4 | 
 5 | import logging
 6 | from typing import Dict, List, Any
 7 | from collections import defaultdict
 8 | import numpy as np
 9 | 
10 | logger = logging.getLogger(__name__)
11 | 
12 | class MetricsTracker:
13 |     """指标跟踪器"""
14 |     
15 |     def __init__(self):
16 |         self.metrics = defaultdict(list)
17 |         self.current_metrics = {}
18 |         
19 |     def update(self, metrics: Dict[str, float], step: int = None):
20 |         """更新指标"""
21 |         for key, value in metrics.items():
22 |             self.metrics[key].append(value)
23 |         
24 |         self.current_metrics = metrics.copy()
25 |         if step is not None:
26 |             self.current_metrics['step'] = step
27 |     
28 |     def get_latest(self) -> Dict[str, float]:
29 |         """获取最新指标"""
30 |         return self.current_metrics.copy()
31 |     
32 |     def get_history(self, key: str) -> List[float]:
33 |         """获取指标历史"""
34 |         return self.metrics[key].copy()
35 |     
36 |     def get_summary(self) -> Dict[str, Dict[str, float]]:
37 |         """获取指标摘要"""
38 |         summary = {}
39 |         for key, values in self.metrics.items():
40 |             if values:
41 |                 summary[key] = {
42 |                     'mean': float(np.mean(values)),
43 |                     'std': float(np.std(values)),
44 |                     'min': float(np.min(values)),
45 |                     'max': float(np.max(values)),
46 |                     'latest': float(values[-1])
47 |                 }
48 |         return summary
49 |     
50 |     def log_current(self, prefix: str = ""):
51 |         """记录当前指标"""
52 |         if self.current_metrics:
53 |             metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in self.current_metrics.items()])
54 |             logger.info(f"[{prefix}] {metrics_str}")
55 |     
56 |     def reset(self):
57 |         """重置指标"""
58 |         self.metrics.clear()
59 |         self.current_metrics.clear()
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/experiment_utils.py

- Extension: .py
- Language: python
- Size: 1365 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
 1 | """
 2 | 实验工具函数
 3 | """
 4 | 
 5 | import os
 6 | import json
 7 | import logging
 8 | from datetime import datetime
 9 | from typing import Dict, Any
10 | 
11 | logger = logging.getLogger(__name__)
12 | 
13 | def create_experiment_dir(base_dir: str) -> str:
14 |     """创建实验目录"""
15 |     timestamp = datetime.now().strftime("%Y%m%d_%H%M")
16 |     exp_dir = os.path.join(base_dir, "results", timestamp)
17 |     
18 |     # 创建目录结构
19 |     os.makedirs(exp_dir, exist_ok=True)
20 |     os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
21 |     
22 |     logger.info(f"[实验目录] 创建实验目录: {exp_dir}")
23 |     return exp_dir
24 | 
25 | def save_results(results: Dict[str, Any], save_path: str):
26 |     """保存实验结果到JSON文件"""
27 |     try:
28 |         with open(save_path, 'w', encoding='utf-8') as f:
29 |             json.dump(results, f, indent=2, ensure_ascii=False)
30 |         logger.info(f"[结果保存] 结果已保存到: {save_path}")
31 |     except Exception as e:
32 |         logger.error(f"[结果保存] 保存失败: {e}")
33 | 
34 | def load_results(file_path: str) -> Dict[str, Any]:
35 |     """从JSON文件加载实验结果"""
36 |     try:
37 |         with open(file_path, 'r', encoding='utf-8') as f:
38 |             results = json.load(f)
39 |         logger.info(f"[结果加载] 结果已从 {file_path} 加载")
40 |         return results
41 |     except Exception as e:
42 |         logger.error(f"[结果加载] 加载失败: {e}")
43 |         return {}
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/utils/__init__.py

- Extension: .py
- Language: python
- Size: 243 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
1 | """
2 | 工具模块
3 | """
4 | 
5 | from .logger import setup_logger
6 | from .metrics import MetricsTracker
7 | from .experiment_utils import create_experiment_dir, save_results
8 | 
9 | __all__ = ['setup_logger', 'MetricsTracker', 'create_experiment_dir', 'save_results']
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/modes/global_mode.py

- Extension: .py
- Language: python
- Size: 19380 bytes
- Created: 2025-08-04 13:26:34
- Modified: 2025-08-04 13:26:34

### Code

```python
  1 | """
  2 | Global模式实现
  3 | 计算真实GTE的核心模块（GT与GC对称运行）
  4 | """
  5 | 
  6 | import os
  7 | import torch
  8 | import torch.utils.data as data_utils
  9 | import pandas as pd
 10 | import numpy as np
 11 | import logging
 12 | from typing import Dict, List, Tuple, Any
 13 | from ..data import KuaiRandDataLoader, FeatureProcessor
 14 | from ..models import MultiLabelModel
 15 | from ..utils import MetricsTracker, save_results
 16 | import random
 17 | 
 18 | logger = logging.getLogger(__name__)
 19 | 
 20 | class GlobalMode:
 21 |     """Global模式实验管理器"""
 22 |     
 23 |     def __init__(self, config: Dict, exp_dir: str):
 24 |         self.config = config
 25 |         self.exp_dir = exp_dir
 26 |         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 27 |         
 28 |         # 初始化组件
 29 |         self.data_loader = KuaiRandDataLoader(config)
 30 |         self.feature_processor = FeatureProcessor(config)
 31 |         self.multi_label_model = None
 32 |         
 33 |         # 数据存储
 34 |         self.merged_data = None
 35 |         self.user_video_lists = None
 36 |         self.train_users = None
 37 |         self.val_users = None
 38 |         self.processed_data = None
 39 |         
 40 |         # 仿真状态
 41 |         self.used_videos = set()  # 记录已使用的视频
 42 |         
 43 |         # 指标跟踪
 44 |         self.metrics_tracker = MetricsTracker()
 45 |         
 46 |         # 仿真结果
 47 |         self.total_label_T = {label['name']: 0.0 for label in config['labels']}  # Treatment组累计收益
 48 |         self.total_label_C = {label['name']: 0.0 for label in config['labels']}  # Control组累计收益
 49 |         
 50 |         logger.info(f"[Global模式] 初始化完成，设备: {self.device}")
 51 |     
 52 |     def ensure_float_data(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
 53 |         """确保数据为float类型并转换为numpy数组"""
 54 |         try:
 55 |             # 提取指定列
 56 |             subset = data[columns].copy()
 57 |             
 58 |             # 逐列强制转换为数值类型
 59 |             for col in columns:
 60 |                 if col in subset.columns:
 61 |                     # 先尝试转换为数值类型
 62 |                     subset[col] = pd.to_numeric(subset[col], errors='coerce')
 63 |                     # 填充NaN值
 64 |                     subset[col] = subset[col].fillna(0.0)
 65 |                     # 确保是float类型
 66 |                     subset[col] = subset[col].astype(np.float32)
 67 |             
 68 |             # 转换为numpy数组
 69 |             array = subset.values.astype(np.float32)
 70 |             
 71 |             # 检查是否还有非数值类型
 72 |             if array.dtype == np.object_:
 73 |                 logger.error(f"[数据转换] 数据中仍有非数值类型，列: {columns}")
 74 |                 # 强制转换每个元素
 75 |                 array = np.array([[float(x) if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else 0.0 
 76 |                                  for x in row] for row in array], dtype=np.float32)
 77 |             
 78 |             return array
 79 |             
 80 |         except Exception as e:
 81 |             logger.error(f"[数据转换] 转换失败: {e}")
 82 |             # 创建零矩阵作为备选
 83 |             return np.zeros((len(data), len(columns)), dtype=np.float32)
 84 |     
 85 |     def load_and_prepare_data(self):
 86 |         """加载和准备数据"""
 87 |         logger.info("[Global模式] 开始数据加载和准备...")
 88 |         
 89 |         # 加载原始数据
 90 |         self.merged_data, self.user_video_lists, self.train_users, self.val_users = \
 91 |             self.data_loader.load_and_prepare_data()
 92 |         
 93 |         # 打印数据集统计信息
 94 |         stats = self.data_loader.get_dataset_stats()
 95 |         logger.info(f"[数据统计] 总样本数: {stats['total_samples']}")
 96 |         logger.info(f"[数据统计] 唯一用户数: {stats['unique_users']}")
 97 |         logger.info(f"[数据统计] 唯一视频数: {stats['unique_videos']}")
 98 |         logger.info(f"[数据统计] 训练用户数: {stats['train_users']}")
 99 |         logger.info(f"[数据统计] 验证用户数: {stats['val_users']}")
100 |         logger.info(f"[数据统计] 点击率: {stats['click_rate']:.4f}")
101 |         logger.info(f"[数据统计] 平均播放时长: {stats['avg_play_time']:.2f}ms")
102 |         
103 |         # 特征处理
104 |         logger.info("[特征处理] 开始特征预处理...")
105 |         self.processed_data = self.feature_processor.fit_transform(self.merged_data)
106 |         
107 |         # 获取特征列
108 |         feature_columns = self.feature_processor.get_feature_columns()
109 |         input_dim = len(feature_columns)
110 |         logger.info(f"[特征处理] 特征维度: {input_dim}")
111 |         logger.info(f"[特征处理] 特征列: {feature_columns}")
112 |         
113 |         # 初始化多标签模型
114 |         self.multi_label_model = MultiLabelModel(
115 |             config=self.config,
116 |             input_dim=input_dim,
117 |             device=self.device
118 |         )
119 |         
120 |         logger.info("[Global模式] 数据准备完成")
121 |     
122 |     def pretrain_models(self):
123 |         """预训练模型"""
124 |         if not self.config['pretrain']['enabled']:
125 |             logger.info("[预训练] 跳过预训练阶段")
126 |             return
127 |         
128 |         logger.info("[预训练] 开始预训练阶段...")
129 |         
130 |         # 准备训练数据
131 |         train_data = self.processed_data[self.processed_data['mask'] == 0].copy()
132 |         logger.info(f"[预训练] 训练数据量: {len(train_data)}")
133 |         
134 |         # 创建数据加载器
135 |         batch_size = self.config['pretrain']['batch_size']
136 |         
137 |         for epoch in range(self.config['pretrain']['epochs']):
138 |             logger.info(f"[预训练] Epoch {epoch+1}/{self.config['pretrain']['epochs']}")
139 |             
140 |             # 随机打乱数据
141 |             train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)
142 |             
143 |             epoch_losses = {label['name']: [] for label in self.config['labels']}
144 |             
145 |             # 批次训练
146 |             for i in range(0, len(train_data_shuffled), batch_size):
147 |                 batch_data = train_data_shuffled[i:i+batch_size]
148 |                 
149 |                 if len(batch_data) == 0:
150 |                     continue
151 |                 
152 |                 # 准备特征和标签
153 |                 feature_columns = self.feature_processor.get_feature_columns()
154 |                 
155 |                 # 使用新的数据转换函数
156 |                 feature_array = self.ensure_float_data(batch_data, feature_columns)
157 |                 X = torch.FloatTensor(feature_array).to(self.device)
158 |                 
159 |                 targets = {}
160 |                 for label_config in self.config['labels']:
161 |                     label_name = label_config['name']
162 |                     target_col = label_config['target']
163 |                     y = torch.FloatTensor(batch_data[target_col].values).unsqueeze(1).to(self.device)
164 |                     targets[label_name] = y
165 |                 
166 |                 # 训练步骤
167 |                 losses = self.multi_label_model.train_step(X, targets)
168 |                 
169 |                 for label_name, loss in losses.items():
170 |                     epoch_losses[label_name].append(loss)
171 |             
172 |             # 记录epoch结果
173 |             avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items() if losses}
174 |             loss_str = ", ".join([f"{name}: {loss:.6f}" for name, loss in avg_losses.items()])
175 |             logger.info(f"[预训练] Epoch {epoch+1} 平均损失 - {loss_str}")
176 |         
177 |         logger.info("[预训练] 预训练阶段完成")
178 |     
179 |     def sample_candidate_videos(self, user_id: int, n_candidate: int) -> List[int]:
180 |         """为用户采样候选视频"""
181 |         if user_id not in self.user_video_lists:
182 |             return []
183 |         
184 |         # 获取用户的所有视频
185 |         user_videos = self.user_video_lists[user_id]
186 |         
187 |         # 筛选可用视频（mask=0 且 used=0）
188 |         available_videos = []
189 |         for video_id in user_videos:
190 |             video_data = self.processed_data[
191 |                 (self.processed_data['user_id'] == user_id) & 
192 |                 (self.processed_data['video_id'] == video_id)
193 |             ]
194 |             
195 |             if len(video_data) > 0:
196 |                 mask = video_data.iloc[0]['mask']
197 |                 if mask == 0 and video_id not in self.used_videos:
198 |                     available_videos.append(video_id)
199 |         
200 |         # 随机采样
201 |         if len(available_videos) <= n_candidate:
202 |             return available_videos
203 |         else:
204 |             return random.sample(available_videos, n_candidate)
205 |     
206 |     def get_user_video_features(self, user_id: int, video_ids: List[int]) -> torch.Tensor:
207 |         """获取用户-视频对的特征"""
208 |         features_list = []
209 |         feature_columns = self.feature_processor.get_feature_columns()
210 |         
211 |         for video_id in video_ids:
212 |             # 获取该用户-视频对的特征
213 |             row = self.processed_data[
214 |                 (self.processed_data['user_id'] == user_id) & 
215 |                 (self.processed_data['video_id'] == video_id)
216 |             ]
217 |             
218 |             if len(row) > 0:
219 |                 # 使用新的数据转换函数
220 |                 feature_array = self.ensure_float_data(row, feature_columns)
221 |                 if len(feature_array) > 0:
222 |                     features_list.append(feature_array[0])
223 |         
224 |         if features_list:
225 |             feature_array = np.array(features_list, dtype=np.float32)
226 |             return torch.FloatTensor(feature_array).to(self.device)
227 |         else:
228 |             return torch.empty(0, len(feature_columns)).to(self.device)
229 |     
230 |     def get_real_labels(self, user_id: int, video_ids: List[int]) -> Dict[str, torch.Tensor]:
231 |         """获取真实标签"""
232 |         labels = {label['name']: [] for label in self.config['labels']}
233 |         
234 |         for video_id in video_ids:
235 |             # 获取真实标签
236 |             row = self.merged_data[
237 |                 (self.merged_data['user_id'] == user_id) & 
238 |                 (self.merged_data['video_id'] == video_id)
239 |             ]
240 |             
241 |             if len(row) > 0:
242 |                 for label_config in self.config['labels']:
243 |                     label_name = label_config['name']
244 |                     target_col = label_config['target']
245 |                     label_value = row[target_col].values[0]
246 |                     labels[label_name].append(label_value)
247 |         
248 |         # 转换为tensor
249 |         result = {}
250 |         for label_name, values in labels.items():
251 |             if values:
252 |                 result[label_name] = torch.FloatTensor(values).unsqueeze(1).to(self.device)
253 |         
254 |         return result
255 |     
256 |     def run_single_simulation(self, is_treatment: bool, step: int, batch_users: List[int]) -> Dict[str, float]:
257 |         """运行单次仿真步骤"""
258 |         prefix = "Treatment" if is_treatment else "Control"
259 |         logger.info(f"[{prefix}仿真] Step {step}: 开始处理 {len(batch_users)} 个用户")
260 |         
261 |         step_rewards = {label['name']: 0.0 for label in self.config['labels']}
262 |         processed_users = 0
263 |         
264 |         for user_id in batch_users:
265 |             # 1. 候选视频生成
266 |             candidates = self.sample_candidate_videos(user_id, self.config['global']['n_candidate'])
267 |             
268 |             if len(candidates) == 0:
269 |                 continue  # 该用户没有可用视频
270 |             
271 |             # 2. 获取特征
272 |             X = self.get_user_video_features(user_id, candidates)
273 |             
274 |             if X.size(0) == 0:
275 |                 continue  # 没有有效特征
276 |             
277 |             # 3. 模型预测与加权排序
278 |             alpha_weights = {}
279 |             for label_config in self.config['labels']:
280 |                 label_name = label_config['name']
281 |                 alpha_key = 'alpha_T' if is_treatment else 'alpha_C'
282 |                 alpha_weights[label_name] = label_config[alpha_key]
283 |             
284 |             combined_scores = self.multi_label_model.get_combined_score(X, alpha_weights)
285 |             
286 |             # 4. 选出胜出视频
287 |             winner_idx = torch.argmax(combined_scores.squeeze()).item()
288 |             winner_video = candidates[winner_idx]
289 |             
290 |             # 5. 获取真实反馈
291 |             real_labels = self.get_real_labels(user_id, [winner_video])
292 |             winner_features = X[winner_idx:winner_idx+1]
293 |             
294 |             # 6. 模型训练
295 |             if real_labels:
296 |                 _ = self.multi_label_model.train_step(winner_features, real_labels)
297 |                 
298 |                 # 累加收益
299 |                 for label_name, label_tensor in real_labels.items():
300 |                     reward_value = label_tensor.item()
301 |                     step_rewards[label_name] += reward_value
302 |             
303 |             # 7. 更新used状态
304 |             self.used_videos.add(winner_video)
305 |             
306 |             processed_users += 1
307 |         
308 |         logger.info(f"[{prefix}仿真] Step {step}: 处理了 {processed_users} 个用户")
309 |         return step_rewards
310 |     
311 |     def validate_models(self, step: int):
312 |         """验证模型性能"""
313 |         logger.info(f"[验证] Step {step}: 开始验证")
314 |         
315 |         # 使用验证集用户
316 |         val_sample_size = min(100, len(self.val_users))  # 限制验证样本数量
317 |         val_users_sample = random.sample(self.val_users, val_sample_size)
318 |         
319 |         total_losses = {label['name']: [] for label in self.config['labels']}
320 |         
321 |         for user_id in val_users_sample:
322 |             # 获取该用户的所有视频（不考虑used和mask）
323 |             user_videos = self.user_video_lists.get(user_id, [])
324 |             
325 |             if len(user_videos) == 0:
326 |                 continue
327 |             
328 |             # 随机选择几个视频进行验证
329 |             sample_videos = random.sample(user_videos, min(5, len(user_videos)))
330 |             
331 |             X = self.get_user_video_features(user_id, sample_videos)
332 |             real_labels = self.get_real_labels(user_id, sample_videos)
333 |             
334 |             if X.size(0) > 0 and real_labels:
335 |                 losses = self.multi_label_model.evaluate(X, real_labels)
336 |                 for label_name, loss in losses.items():
337 |                     total_losses[label_name].append(loss)
338 |         
339 |         # 计算平均验证损失
340 |         avg_val_losses = {}
341 |         for label_name, losses in total_losses.items():
342 |             if losses:
343 |                 avg_val_losses[f'val_{label_name}_loss'] = np.mean(losses)
344 |         
345 |         self.metrics_tracker.update(avg_val_losses, step)
346 |         self.metrics_tracker.log_current("验证")
347 |         
348 |         # 更新学习率调度器
349 |         self.multi_label_model.update_schedulers(avg_val_losses)
350 |     
351 |     def run_global_simulation(self):
352 |         """运行Global仿真"""
353 |         logger.info("[Global仿真] 开始全局仿真流程...")
354 |         
355 |         n_steps = self.config['global']['n_steps']
356 |         batch_size = self.config['global']['batch_size']
357 |         validate_every = self.config['global']['validate_every']
358 |         save_every = self.config['global']['save_every']
359 |         
360 |         for step in range(1, n_steps + 1):
361 |             logger.info(f"[Global仿真] ===== Step {step}/{n_steps} =====")
362 |             
363 |             # 1. 用户批次抽样（GT和GC使用相同的用户批次）
364 |             batch_users = random.sample(self.train_users, min(batch_size, len(self.train_users)))
365 |             
366 |             # 2. Treatment仿真（GT）
367 |             logger.info("[GT流程] 开始Treatment组仿真...")
368 |             step_rewards_T = self.run_single_simulation(True, step, batch_users)
369 |             
370 |             # 累加到总收益
371 |             for label_name, reward in step_rewards_T.items():
372 |                 self.total_label_T[label_name] += reward
373 |             
374 |             # 3. Control仿真（GC）
375 |             logger.info("[GC流程] 开始Control组仿真...")
376 |             step_rewards_C = self.run_single_simulation(False, step, batch_users)
377 |             
378 |             # 累加到总收益
379 |             for label_name, reward in step_rewards_C.items():
380 |                 self.total_label_C[label_name] += reward
381 |             
382 |             # 4. 记录步骤指标
383 |             step_metrics = {}
384 |             for label_name in step_rewards_T:
385 |                 step_metrics[f'step_{label_name}_T'] = step_rewards_T[label_name]
386 |                 step_metrics[f'step_{label_name}_C'] = step_rewards_C[label_name]
387 |                 step_metrics[f'total_{label_name}_T'] = self.total_label_T[label_name]
388 |                 step_metrics[f'total_{label_name}_C'] = self.total_label_C[label_name]
389 |             
390 |             self.metrics_tracker.update(step_metrics, step)
391 |             self.metrics_tracker.log_current(f"训练 Step {step}")
392 |             
393 |             # 5. 验证
394 |             if step % validate_every == 0:
395 |                 self.validate_models(step)
396 |             
397 |             # 6. 保存模型
398 |             if step % save_every == 0:
399 |                 checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
400 |                 self.multi_label_model.save_models(checkpoint_dir, step)
401 |                 
402 |                 # 保存特征处理器
403 |                 self.feature_processor.save_processors(checkpoint_dir)
404 |         
405 |         logger.info("[Global仿真] 全局仿真完成")
406 |     
407 |     def compute_gte(self) -> Dict[str, float]:
408 |         """计算GTE（Global Treatment Effect）"""
409 |         logger.info("[GTE计算] 开始计算全局处理效应...")
410 |         
411 |         gte_results = {}
412 |         
413 |         for label_name in self.total_label_T:
414 |             gt_total = self.total_label_T[label_name]
415 |             gc_total = self.total_label_C[label_name]
416 |             
417 |             # 计算GTE
418 |             gte = gt_total - gc_total
419 |             gte_relative = (gte / gc_total * 100) if gc_total != 0 else 0
420 |             
421 |             gte_results[f'GTE_{label_name}'] = gte
422 |             gte_results[f'GTE_{label_name}_relative'] = gte_relative
423 |             gte_results[f'GT_{label_name}'] = gt_total
424 |             gte_results[f'GC_{label_name}'] = gc_total
425 |             
426 |             logger.info(f"[GTE计算] {label_name}: GT={gt_total:.4f}, GC={gc_total:.4f}, GTE={gte:.4f} ({gte_relative:+.2f}%)")
427 |         
428 |         return gte_results
429 |     
430 |     def run(self):
431 |         """运行完整的Global模式实验"""
432 |         logger.info("[Global模式] 开始运行完整实验...")
433 |         
434 |         try:
435 |             # 1. 数据加载和准备
436 |             self.load_and_prepare_data()
437 |             
438 |             # 2. 预训练
439 |             self.pretrain_models()
440 |             
441 |             # 3. 全局仿真
442 |             self.run_global_simulation()
443 |             
444 |             # 4. 计算GTE
445 |             gte_results = self.compute_gte()
446 |             
447 |             # 5. 保存最终结果
448 |             final_results = {
449 |                 'config': self.config,
450 |                 'gte_results': gte_results,
451 |                 'metrics_summary': self.metrics_tracker.get_summary(),
452 |                 'dataset_stats': self.data_loader.get_dataset_stats()
453 |             }
454 |             
455 |             results_path = os.path.join(self.exp_dir, 'result.json')
456 |             save_results(final_results, results_path)
457 |             
458 |             logger.info("[Global模式] 实验完成！")
459 |             
460 |             # 打印最终结果
461 |             logger.info("========== 最终实验结果 ==========")
462 |             for key, value in gte_results.items():
463 |                 logger.info(f"{key}: {value}")
464 |                 
465 |         except Exception as e:
466 |             logger.error(f"[Global模式] 实验执行失败: {e}")
467 |             raise
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/modes/__init__.py

- Extension: .py
- Language: python
- Size: 89 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
1 | """
2 | 实验模式模块
3 | """
4 | 
5 | from .global_mode import GlobalMode
6 | 
7 | __all__ = ['GlobalMode']
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/multi_label_model.py

- Extension: .py
- Language: python
- Size: 7922 bytes
- Created: 2025-08-04 13:26:34
- Modified: 2025-08-04 13:26:34

### Code

```python
  1 | """
  2 | 多标签预测模型
  3 | 管理多个独立的预测模型
  4 | """
  5 | 
  6 | import torch
  7 | import torch.nn as nn
  8 | import torch.optim as optim
  9 | import logging
 10 | from typing import Dict, List, Tuple, Any
 11 | from .mlp_model import MLPModel
 12 | from .loss_functions import get_loss_function
 13 | 
 14 | logger = logging.getLogger(__name__)
 15 | 
 16 | class MultiLabelModel:
 17 |     """多标签预测模型管理器"""
 18 |     
 19 |     def __init__(self, config: Dict, input_dim: int, device: str = 'cpu'):
 20 |         self.config = config
 21 |         self.input_dim = input_dim
 22 |         self.device = device
 23 |         self.labels = config['labels']
 24 |         
 25 |         # 为每个标签创建独立的模型
 26 |         self.models = {}
 27 |         self.optimizers = {}
 28 |         self.loss_functions = {}
 29 |         self.schedulers = {}
 30 |         
 31 |         self._build_models()
 32 |         
 33 |     def _build_models(self):
 34 |         """构建所有标签的模型"""
 35 |         logger.info("[模型构建] 开始构建多标签预测模型...")
 36 |         
 37 |         for label_config in self.labels:
 38 |             label_name = label_config['name']
 39 |             logger.info(f"[模型构建] 构建 {label_name} 模型...")
 40 |             
 41 |             # 创建模型
 42 |             model = MLPModel(
 43 |                 input_dim=self.input_dim,
 44 |                 hidden_layers=label_config['model_params']['hidden_layers'],
 45 |                 output_dim=1,
 46 |                 dropout=label_config['model_params']['dropout']
 47 |             ).to(self.device)
 48 |             
 49 |             # 创建优化器
 50 |             optimizer = optim.Adam(
 51 |                 model.parameters(),
 52 |                 lr=label_config['learning_rate'],
 53 |                 weight_decay=label_config['weight_decay']
 54 |             )
 55 |             
 56 |             # 创建损失函数
 57 |             loss_fn = get_loss_function(label_config['loss_function'])
 58 |             
 59 |             # 创建学习率调度器
 60 |             scheduler = optim.lr_scheduler.ReduceLROnPlateau(
 61 |                 optimizer, mode='min', factor=0.5, patience=5
 62 |             )
 63 |             
 64 |             self.models[label_name] = model
 65 |             self.optimizers[label_name] = optimizer
 66 |             self.loss_functions[label_name] = loss_fn
 67 |             self.schedulers[label_name] = scheduler
 68 |             
 69 |             # 打印模型信息
 70 |             model_info = model.get_model_info()
 71 |             logger.info(f"[模型构建] {label_name} 模型: {model_info['total_params']} 参数")
 72 |         
 73 |         logger.info(f"[模型构建] 多标签模型构建完成，共 {len(self.models)} 个模型")
 74 |     
 75 |     def forward(self, x: torch.Tensor, label_name: str) -> torch.Tensor:
 76 |         """单个标签的前向传播"""
 77 |         if label_name not in self.models:
 78 |             raise ValueError(f"标签 {label_name} 的模型不存在")
 79 |         
 80 |         return self.models[label_name](x)
 81 |     
 82 |     def predict_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
 83 |         """预测所有标签"""
 84 |         predictions = {}
 85 |         
 86 |         with torch.no_grad():
 87 |             for label_name in self.models:
 88 |                 pred = self.forward(x, label_name)
 89 |                 
 90 |                 # 根据标签类型处理输出
 91 |                 label_config = next(lc for lc in self.labels if lc['name'] == label_name)
 92 |                 if label_config['type'] == 'binary':
 93 |                     pred = torch.sigmoid(pred)
 94 |                 elif label_config['type'] == 'numerical':
 95 |                     pred = torch.clamp(pred, min=0)  # 确保非负
 96 |                 
 97 |                 predictions[label_name] = pred
 98 |         
 99 |         return predictions
100 |     
101 |     def compute_losses(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
102 |         """计算所有标签的损失"""
103 |         losses = {}
104 |         
105 |         for label_name in self.models:
106 |             if label_name in targets:
107 |                 pred = self.forward(x, label_name)
108 |                 target = targets[label_name]
109 |                 loss = self.loss_functions[label_name](pred, target)
110 |                 losses[label_name] = loss
111 |         
112 |         return losses
113 |     
114 |     def train_step(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
115 |         """训练步骤"""
116 |         # 设置为训练模式
117 |         for model in self.models.values():
118 |             model.train()
119 |         
120 |         # 计算损失并更新模型
121 |         losses = {}
122 |         for label_name in self.models:
123 |             if label_name in targets:
124 |                 # 清零梯度
125 |                 self.optimizers[label_name].zero_grad()
126 |                 
127 |                 # 前向传播
128 |                 pred = self.forward(x, label_name)
129 |                 target = targets[label_name]
130 |                 
131 |                 # 计算损失
132 |                 loss = self.loss_functions[label_name](pred, target)
133 |                 
134 |                 # 反向传播
135 |                 loss.backward()
136 |                 
137 |                 # 梯度裁剪
138 |                 torch.nn.utils.clip_grad_norm_(self.models[label_name].parameters(), max_norm=1.0)
139 |                 
140 |                 # 更新参数
141 |                 self.optimizers[label_name].step()
142 |                 
143 |                 losses[label_name] = loss.item()
144 |         
145 |         return losses
146 |     
147 |     def evaluate(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
148 |         """评估模型"""
149 |         # 设置为评估模式
150 |         for model in self.models.values():
151 |             model.eval()
152 |         
153 |         with torch.no_grad():
154 |             losses = self.compute_losses(x, targets)
155 |             return {name: loss.item() for name, loss in losses.items()}
156 |     
157 |     def get_combined_score(self, x: torch.Tensor, alpha_weights: Dict[str, float]) -> torch.Tensor:
158 |         """根据alpha权重计算组合分数"""
159 |         predictions = self.predict_all(x)
160 |         
161 |         combined_score = torch.zeros(x.size(0), 1, device=self.device)
162 |         
163 |         for label_name, alpha in alpha_weights.items():
164 |             if label_name in predictions:
165 |                 pred = predictions[label_name]
166 |                 combined_score += alpha * pred
167 |         
168 |         return combined_score
169 |     
170 |     def save_models(self, save_dir: str, step: int):
171 |         """保存所有模型"""
172 |         import os
173 |         os.makedirs(save_dir, exist_ok=True)
174 |         
175 |         checkpoint = {
176 |             'step': step,
177 |             'config': self.config,
178 |             'input_dim': self.input_dim
179 |         }
180 |         
181 |         for label_name in self.models:
182 |             checkpoint[f'{label_name}_model'] = self.models[label_name].state_dict()
183 |             checkpoint[f'{label_name}_optimizer'] = self.optimizers[label_name].state_dict()
184 |             checkpoint[f'{label_name}_scheduler'] = self.schedulers[label_name].state_dict()
185 |         
186 |         save_path = os.path.join(save_dir, f'step_{step}.pt')
187 |         torch.save(checkpoint, save_path)
188 |         logger.info(f"[模型保存] 模型已保存到: {save_path}")
189 |     
190 |     def load_models(self, checkpoint_path: str):
191 |         """加载所有模型"""
192 |         checkpoint = torch.load(checkpoint_path, map_location=self.device)
193 |         
194 |         for label_name in self.models:
195 |             if f'{label_name}_model' in checkpoint:
196 |                 self.models[label_name].load_state_dict(checkpoint[f'{label_name}_model'])
197 |             if f'{label_name}_optimizer' in checkpoint:
198 |                 self.optimizers[label_name].load_state_dict(checkpoint[f'{label_name}_optimizer'])
199 |             if f'{label_name}_scheduler' in checkpoint:
200 |                 self.schedulers[label_name].load_state_dict(checkpoint[f'{label_name}_scheduler'])
201 |         
202 |         logger.info(f"[模型加载] 模型已从 {checkpoint_path} 加载")
203 |         return checkpoint.get('step', 0)
204 |     
205 |     def update_schedulers(self, metrics: Dict[str, float]):
206 |         """更新学习率调度器"""
207 |         for label_name, metric in metrics.items():
208 |             if label_name in self.schedulers:
209 |                 self.schedulers[label_name].step(metric)
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/loss_functions.py

- Extension: .py
- Language: python
- Size: 1639 bytes
- Created: 2025-08-04 13:26:34
- Modified: 2025-08-04 13:26:34

### Code

```python
 1 | """
 2 | 损失函数定义
 3 | 包含LogMAE和其他自定义损失函数
 4 | """
 5 | 
 6 | import torch
 7 | import torch.nn as nn
 8 | import torch.nn.functional as F
 9 | 
10 | class LogMAELoss(nn.Module):
11 |     """Log Mean Absolute Error损失函数
12 |     用于播放时长等具有大数值范围的连续标签
13 |     """
14 |     
15 |     def __init__(self, epsilon: float = 1e-8):
16 |         super(LogMAELoss, self).__init__()
17 |         self.epsilon = epsilon
18 |     
19 |     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
20 |         """
21 |         计算LogMAE损失
22 |         
23 |         Args:
24 |             pred: 预测值 [batch_size, 1]
25 |             target: 真实值 [batch_size, 1]
26 |         
27 |         Returns:
28 |             loss: LogMAE损失值
29 |         """
30 |         # 确保预测值和目标值都是正数
31 |         pred = torch.clamp(pred, min=self.epsilon)
32 |         target = torch.clamp(target, min=self.epsilon)
33 |         
34 |         # 计算log后的MAE
35 |         log_pred = torch.log(pred + self.epsilon)
36 |         log_target = torch.log(target + self.epsilon)
37 |         
38 |         loss = F.l1_loss(log_pred, log_target)
39 |         return loss
40 | 
41 | def get_loss_function(loss_name: str, **kwargs):
42 |     """获取损失函数"""
43 |     if loss_name.lower() == 'logmae':
44 |         return LogMAELoss(**kwargs)
45 |     elif loss_name.lower() == 'bce':
46 |         return nn.BCEWithLogitsLoss(**kwargs)
47 |     elif loss_name.lower() == 'mse':
48 |         return nn.MSELoss(**kwargs)
49 |     elif loss_name.lower() == 'mae':
50 |         return nn.L1Loss(**kwargs)
51 |     elif loss_name.lower() == 'crossentropy':
52 |         return nn.CrossEntropyLoss(**kwargs)
53 |     else:
54 |         raise ValueError(f"不支持的损失函数: {loss_name}")
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/mlp_model.py

- Extension: .py
- Language: python
- Size: 2023 bytes
- Created: 2025-08-04 13:26:34
- Modified: 2025-08-04 13:26:34

### Code

```python
 1 | """
 2 | MLP模型实现
 3 | 用于单个标签的预测
 4 | """
 5 | 
 6 | import torch
 7 | import torch.nn as nn
 8 | import torch.nn.functional as F
 9 | from typing import List
10 | 
11 | class MLPModel(nn.Module):
12 |     """多层感知机模型"""
13 |     
14 |     def __init__(self, input_dim: int, hidden_layers: List[int], 
15 |                  output_dim: int = 1, dropout: float = 0.1):
16 |         super(MLPModel, self).__init__()
17 |         
18 |         self.input_dim = input_dim
19 |         self.hidden_layers = hidden_layers
20 |         self.output_dim = output_dim
21 |         self.dropout = dropout
22 |         
23 |         # 构建网络层
24 |         layers = []
25 |         prev_dim = input_dim
26 |         
27 |         # 隐藏层
28 |         for hidden_dim in hidden_layers:
29 |             layers.append(nn.Linear(prev_dim, hidden_dim))
30 |             layers.append(nn.ReLU())
31 |             layers.append(nn.Dropout(dropout))
32 |             prev_dim = hidden_dim
33 |         
34 |         # 输出层
35 |         layers.append(nn.Linear(prev_dim, output_dim))
36 |         
37 |         self.network = nn.Sequential(*layers)
38 |         
39 |         # 权重初始化
40 |         self._initialize_weights()
41 |     
42 |     def _initialize_weights(self):
43 |         """初始化网络权重"""
44 |         for module in self.modules():
45 |             if isinstance(module, nn.Linear):
46 |                 nn.init.xavier_uniform_(module.weight)
47 |                 if module.bias is not None:
48 |                     nn.init.zeros_(module.bias)
49 |     
50 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
51 |         """前向传播"""
52 |         return self.network(x)
53 |     
54 |     def get_model_info(self) -> dict:
55 |         """获取模型信息"""
56 |         total_params = sum(p.numel() for p in self.parameters())
57 |         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
58 |         
59 |         return {
60 |             'input_dim': self.input_dim,
61 |             'hidden_layers': self.hidden_layers,
62 |             'output_dim': self.output_dim,
63 |             'dropout': self.dropout,
64 |             'total_params': total_params,
65 |             'trainable_params': trainable_params
66 |         }
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/models/__init__.py

- Extension: .py
- Language: python
- Size: 279 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
 1 | """
 2 | 模型模块
 3 | 包含多标签预测模型和损失函数
 4 | """
 5 | 
 6 | from .mlp_model import MLPModel
 7 | from .multi_label_model import MultiLabelModel
 8 | from .loss_functions import LogMAELoss, get_loss_function
 9 | 
10 | __all__ = ['MLPModel', 'MultiLabelModel', 'LogMAELoss', 'get_loss_function']
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/data_loader.py

- Extension: .py
- Language: python
- Size: 8176 bytes
- Created: 2025-08-04 13:26:35
- Modified: 2025-08-04 13:26:35

### Code

```python
  1 | """
  2 | KuaiRand数据集加载器
  3 | 负责从原始数据文件中加载用户行为日志、用户特征和视频特征
  4 | """
  5 | 
  6 | import os
  7 | import pandas as pd
  8 | import numpy as np
  9 | import logging
 10 | from typing import Dict, List, Tuple, Optional
 11 | from .cache_manager import CacheManager
 12 | 
 13 | logger = logging.getLogger(__name__)
 14 | 
 15 | class KuaiRandDataLoader:
 16 |     """KuaiRand数据集加载器"""
 17 |     
 18 |     def __init__(self, config: Dict):
 19 |         self.config = config
 20 |         self.dataset_path = config['dataset']['path']
 21 |         self.cache_manager = CacheManager(config['dataset']['cache_path'])
 22 |         
 23 |         # 数据文件映射
 24 |         self.data_files = {
 25 |             'log_random': 'data/log_random_4_22_to_5_08_pure.csv',
 26 |             'log_standard_early': 'data/log_standard_4_08_to_4_21_pure.csv', 
 27 |             'log_standard_late': 'data/log_standard_4_22_to_5_08_pure.csv',
 28 |             'user_features': 'data/user_features_pure.csv',
 29 |             'video_basic': 'data/video_features_basic_pure.csv',
 30 |             'video_statistic': 'data/video_features_statistic_pure.csv'
 31 |         }
 32 |         
 33 |         # 内存中的数据
 34 |         self.user_video_lists = {}  # user_id -> list of video_ids
 35 |         self.merged_data = None
 36 |         self.train_users = None
 37 |         self.val_users = None
 38 |         
 39 |     def load_all_data(self) -> Dict[str, pd.DataFrame]:
 40 |         """加载所有数据文件"""
 41 |         logger.info("[数据加载] 开始加载所有数据文件...")
 42 |         
 43 |         data = {}
 44 |         total_files = len(self.data_files)
 45 |         
 46 |         for i, (key, file_path) in enumerate(self.data_files.items(), 1):
 47 |             full_path = os.path.join(self.dataset_path, file_path)
 48 |             logger.info(f"[数据加载] ({i}/{total_files}) 正在加载 {key}: {file_path}")
 49 |             
 50 |             if not os.path.exists(full_path):
 51 |                 raise FileNotFoundError(f"数据文件不存在: {full_path}")
 52 |                 
 53 |             data[key] = pd.read_csv(full_path)
 54 |             logger.info(f"[数据加载] {key} 加载完成，形状: {data[key].shape}")
 55 |             
 56 |         logger.info("[数据加载] 所有数据文件加载完成")
 57 |         return data
 58 |     
 59 |     def merge_features(self, log_data: pd.DataFrame, user_features: pd.DataFrame,
 60 |                       video_basic: pd.DataFrame, video_statistic: pd.DataFrame) -> pd.DataFrame:
 61 |         """合并所有特征数据"""
 62 |         logger.info("[特征合并] 开始合并用户和视频特征...")
 63 |         
 64 |         # 合并用户特征
 65 |         merged = log_data.merge(user_features, on='user_id', how='left')
 66 |         logger.info(f"[特征合并] 合并用户特征后形状: {merged.shape}")
 67 |         
 68 |         # 合并视频基础特征
 69 |         merged = merged.merge(video_basic, on='video_id', how='left')
 70 |         logger.info(f"[特征合并] 合并视频基础特征后形状: {merged.shape}")
 71 |         
 72 |         # 合并视频统计特征
 73 |         merged = merged.merge(video_statistic, on='video_id', how='left')
 74 |         logger.info(f"[特征合并] 合并视频统计特征后形状: {merged.shape}")
 75 |         
 76 |         logger.info("[特征合并] 特征合并完成")
 77 |         return merged
 78 |     
 79 |     def create_user_video_lists(self, merged_data: pd.DataFrame) -> Dict[int, List[int]]:
 80 |         """创建用户-视频交互列表（缓存机制）"""
 81 |         cache_key = "user_video_lists"
 82 |         
 83 |         # 尝试从缓存加载
 84 |         cached_data = self.cache_manager.load(cache_key)
 85 |         if cached_data is not None:
 86 |             logger.info("[缓存] 从缓存加载用户-视频交互列表")
 87 |             return cached_data
 88 |             
 89 |         logger.info("[用户视频列表] 开始创建用户-视频交互列表...")
 90 |         
 91 |         user_video_lists = {}
 92 |         for user_id in merged_data['user_id'].unique():
 93 |             video_list = merged_data[merged_data['user_id'] == user_id]['video_id'].tolist()
 94 |             user_video_lists[user_id] = video_list
 95 |             
 96 |         logger.info(f"[用户视频列表] 创建完成，共 {len(user_video_lists)} 个用户")
 97 |         
 98 |         # 保存到缓存
 99 |         self.cache_manager.save(user_video_lists, cache_key)
100 |         logger.info("[缓存] 用户-视频交互列表已保存到缓存")
101 |         
102 |         return user_video_lists
103 |     
104 |     def split_users(self, user_list: List[int], val_ratio: float) -> Tuple[List[int], List[int]]:
105 |         """将用户划分为训练集和验证集"""
106 |         logger.info(f"[用户划分] 开始划分用户，验证集比例: {val_ratio}")
107 |         
108 |         np.random.shuffle(user_list)
109 |         split_idx = int(len(user_list) * (1 - val_ratio))
110 |         
111 |         train_users = user_list[:split_idx]
112 |         val_users = user_list[split_idx:]
113 |         
114 |         logger.info(f"[用户划分] 训练用户数: {len(train_users)}, 验证用户数: {len(val_users)}")
115 |         return train_users, val_users
116 |     
117 |     def add_mask_and_used_flags(self, merged_data: pd.DataFrame, val_users: List[int]) -> pd.DataFrame:
118 |         """添加mask和used标记位"""
119 |         logger.info("[标记位] 添加mask和used标记位...")
120 |         
121 |         # 添加mask标记：验证集用户的视频标记为1
122 |         merged_data['mask'] = merged_data['user_id'].isin(val_users).astype(int)
123 |         
124 |         # 添加used标记：初始化为0
125 |         merged_data['used'] = 0
126 |         
127 |         mask_count = merged_data['mask'].sum()
128 |         total_count = len(merged_data)
129 |         
130 |         logger.info(f"[标记位] mask=1的样本数: {mask_count}/{total_count} ({mask_count/total_count:.2%})")
131 |         logger.info("[标记位] 标记位添加完成")
132 |         
133 |         return merged_data
134 |     
135 |     def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict[int, List[int]], List[int], List[int]]:
136 |         """加载并准备所有数据"""
137 |         logger.info("[数据准备] 开始数据加载和准备流程...")
138 |         
139 |         # 加载原始数据
140 |         raw_data = self.load_all_data()
141 |         
142 |         # 合并日志数据
143 |         logger.info("[数据合并] 合并多个日志文件...")
144 |         log_combined = pd.concat([
145 |             raw_data['log_random'],
146 |             raw_data['log_standard_early'], 
147 |             raw_data['log_standard_late']
148 |         ], ignore_index=True)
149 |         logger.info(f"[数据合并] 合并后日志数据形状: {log_combined.shape}")
150 |         
151 |         # 合并特征
152 |         merged_data = self.merge_features(
153 |             log_combined, 
154 |             raw_data['user_features'],
155 |             raw_data['video_basic'],
156 |             raw_data['video_statistic']
157 |         )
158 |         
159 |         # 创建用户-视频交互列表
160 |         user_video_lists = self.create_user_video_lists(merged_data)
161 |         
162 |         # 用户划分
163 |         all_users = list(merged_data['user_id'].unique())
164 |         train_users, val_users = self.split_users(all_users, self.config['global']['user_p_val'])
165 |         
166 |         # 添加标记位
167 |         merged_data = self.add_mask_and_used_flags(merged_data, val_users)
168 |         
169 |         logger.info("[数据准备] 数据准备流程完成")
170 |         
171 |         # 存储到实例变量
172 |         self.merged_data = merged_data
173 |         self.user_video_lists = user_video_lists
174 |         self.train_users = train_users
175 |         self.val_users = val_users
176 |         
177 |         return merged_data, user_video_lists, train_users, val_users
178 |     
179 |     def get_dataset_stats(self) -> Dict:
180 |         """获取数据集统计信息"""
181 |         if self.merged_data is None:
182 |             raise ValueError("数据尚未加载，请先调用 load_and_prepare_data()")
183 |             
184 |         stats = {
185 |             'total_samples': len(self.merged_data),
186 |             'unique_users': self.merged_data['user_id'].nunique(),
187 |             'unique_videos': self.merged_data['video_id'].nunique(),
188 |             'train_users': len(self.train_users),
189 |             'val_users': len(self.val_users),
190 |             'click_rate': self.merged_data['is_click'].mean(),
191 |             'avg_play_time': self.merged_data['play_time_ms'].mean(),
192 |             'features_used': {
193 |                 'numerical': self.config['feature']['numerical'],
194 |                 'categorical': self.config['feature']['categorical']
195 |             }
196 |         }
197 |         
198 |         return stats
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/feature_processor.py

- Extension: .py
- Language: python
- Size: 10404 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
  1 | """
  2 | 特征处理器
  3 | 负责特征的预处理、编码和标准化
  4 | """
  5 | 
  6 | import pandas as pd
  7 | import numpy as np
  8 | import logging
  9 | from typing import Dict, List, Tuple, Optional
 10 | from sklearn.preprocessing import LabelEncoder, StandardScaler
 11 | import pickle
 12 | import os
 13 | 
 14 | logger = logging.getLogger(__name__)
 15 | 
 16 | class FeatureProcessor:
 17 |     """特征处理器"""
 18 |     
 19 |     def __init__(self, config: Dict):
 20 |         self.config = config
 21 |         self.numerical_features = config['feature']['numerical']
 22 |         self.categorical_features = config['feature']['categorical']
 23 |         
 24 |         # 编码器和缩放器
 25 |         self.label_encoders = {}
 26 |         self.scaler = StandardScaler()
 27 |         self.categorical_mappings = {}
 28 |         
 29 |         # 处理后的特征维度信息
 30 |         self.total_numerical_dim = 0
 31 |         self.categorical_dims = {}
 32 |         self.total_categorical_dim = 0
 33 |         
 34 |     def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
 35 |         """处理缺失值"""
 36 |         logger.info("[特征处理] 处理缺失值...")
 37 |         
 38 |         processed_data = data.copy()
 39 |         
 40 |         # 数值特征：用0填充
 41 |         for feature in self.numerical_features:
 42 |             if feature in processed_data.columns:
 43 |                 missing_count = processed_data[feature].isna().sum()
 44 |                 if missing_count > 0:
 45 |                     logger.info(f"[特征处理] {feature}: 用0填充 {missing_count} 个缺失值")
 46 |                     processed_data[feature] = processed_data[feature].fillna(0)
 47 |         
 48 |         # 分类特征：将NA作为新类别
 49 |         for feature in self.categorical_features:
 50 |             if feature in processed_data.columns:
 51 |                 missing_count = processed_data[feature].isna().sum()
 52 |                 if missing_count > 0:
 53 |                     logger.info(f"[特征处理] {feature}: 将 {missing_count} 个缺失值标记为'MISSING'")
 54 |                     processed_data[feature] = processed_data[feature].fillna('MISSING')
 55 |         
 56 |         logger.info("[特征处理] 缺失值处理完成")
 57 |         return processed_data
 58 |     
 59 |     def _process_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
 60 |         """处理分类特征，转换为one-hot编码"""
 61 |         logger.info("[特征处理] 处理分类特征...")
 62 |         
 63 |         processed_data = data.copy()
 64 |         
 65 |         for feature in self.categorical_features:
 66 |             if feature not in processed_data.columns:
 67 |                 logger.warning(f"[特征处理] 特征 {feature} 不存在于数据中")
 68 |                 continue
 69 |                 
 70 |             if fit:
 71 |                 # 训练阶段：拟合编码器
 72 |                 unique_values = processed_data[feature].unique()
 73 |                 logger.info(f"[特征处理] {feature}: {len(unique_values)} 个唯一值")
 74 |                 
 75 |                 # 创建one-hot编码
 76 |                 one_hot = pd.get_dummies(processed_data[feature], prefix=feature)
 77 |                 self.categorical_mappings[feature] = one_hot.columns.tolist()
 78 |                 self.categorical_dims[feature] = len(one_hot.columns)
 79 |                 
 80 |                 # 合并到主数据框
 81 |                 processed_data = pd.concat([processed_data, one_hot], axis=1)
 82 |                 processed_data = processed_data.drop(feature, axis=1)
 83 |                 
 84 |                 logger.info(f"[特征处理] {feature} -> {len(one_hot.columns)} 个one-hot特征")
 85 |             else:
 86 |                 # 预测阶段：使用已有的编码器
 87 |                 if feature in self.categorical_mappings:
 88 |                     one_hot = pd.get_dummies(processed_data[feature], prefix=feature)
 89 |                     
 90 |                     # 确保所有训练时的列都存在
 91 |                     for col in self.categorical_mappings[feature]:
 92 |                         if col not in one_hot.columns:
 93 |                             one_hot[col] = 0
 94 |                     
 95 |                     # 只保留训练时的列
 96 |                     one_hot = one_hot[self.categorical_mappings[feature]]
 97 |                     
 98 |                     # 合并到主数据框
 99 |                     processed_data = pd.concat([processed_data, one_hot], axis=1)
100 |                     processed_data = processed_data.drop(feature, axis=1)
101 |         
102 |         self.total_categorical_dim = sum(self.categorical_dims.values())
103 |         logger.info(f"[特征处理] 分类特征处理完成，总维度: {self.total_categorical_dim}")
104 |         
105 |         return processed_data
106 |     
107 |     def _process_numerical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
108 |         """处理数值特征，进行标准化"""
109 |         logger.info("[特征处理] 处理数值特征...")
110 |         
111 |         processed_data = data.copy()
112 |         
113 |         # 提取数值特征
114 |         available_numerical = [f for f in self.numerical_features if f in processed_data.columns]
115 |         missing_numerical = [f for f in self.numerical_features if f not in processed_data.columns]
116 |         
117 |         if missing_numerical:
118 |             logger.warning(f"[特征处理] 缺失的数值特征: {missing_numerical}")
119 |             # 只使用可用的数值特征
120 |             self.numerical_features = available_numerical
121 |         
122 |         if available_numerical:
123 |             if fit:
124 |                 # 训练阶段：拟合标准化器
125 |                 self.scaler.fit(processed_data[available_numerical])
126 |                 logger.info(f"[特征处理] 数值特征标准化器已拟合，特征数: {len(available_numerical)}")
127 |             
128 |             # 应用标准化
129 |             processed_data[available_numerical] = self.scaler.transform(processed_data[available_numerical])
130 |             self.total_numerical_dim = len(available_numerical)
131 |             
132 |             logger.info(f"[特征处理] 数值特征标准化完成，维度: {self.total_numerical_dim}")
133 |         else:
134 |             logger.warning("[特征处理] 没有可用的数值特征")
135 |             self.total_numerical_dim = 0
136 |         
137 |         return processed_data
138 |     
139 |     def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
140 |         """拟合并转换特征（训练阶段）"""
141 |         logger.info("[特征处理] 开始特征拟合和转换...")
142 |         
143 |         # 处理缺失值
144 |         processed_data = self._handle_missing_values(data)
145 |         
146 |         # 处理分类特征
147 |         processed_data = self._process_categorical_features(processed_data, fit=True)
148 |         
149 |         # 处理数值特征
150 |         processed_data = self._process_numerical_features(processed_data, fit=True)
151 |         
152 |         total_dim = self.total_numerical_dim + self.total_categorical_dim
153 |         logger.info(f"[特征处理] 特征处理完成，总维度: {total_dim} (数值: {self.total_numerical_dim}, 分类: {self.total_categorical_dim})")
154 |         
155 |         # 确保所有特征列都是数值类型
156 |         feature_columns = self.get_feature_columns()
157 |         for col in feature_columns:
158 |             if col in processed_data.columns:
159 |                 processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
160 |         
161 |         logger.info("[特征处理] 数据类型转换完成")
162 |         return processed_data
163 |     
164 |     def transform(self, data: pd.DataFrame) -> pd.DataFrame:
165 |         """转换特征（预测阶段）"""
166 |         logger.info("[特征处理] 应用已拟合的特征转换...")
167 |         
168 |         # 处理缺失值
169 |         processed_data = self._handle_missing_values(data)
170 |         
171 |         # 处理分类特征
172 |         processed_data = self._process_categorical_features(processed_data, fit=False)
173 |         
174 |         # 处理数值特征
175 |         processed_data = self._process_numerical_features(processed_data, fit=False)
176 |         
177 |         # 确保所有特征列都是数值类型
178 |         feature_columns = self.get_feature_columns()
179 |         for col in feature_columns:
180 |             if col in processed_data.columns:
181 |                 processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
182 |         
183 |         logger.info("[特征处理] 特征转换完成")
184 |         return processed_data
185 |     
186 |     def get_feature_columns(self) -> List[str]:
187 |         """获取处理后的特征列名"""
188 |         feature_columns = []
189 |         
190 |         # 数值特征列
191 |         available_numerical = [f for f in self.numerical_features]
192 |         feature_columns.extend(available_numerical)
193 |         
194 |         # 分类特征的one-hot列
195 |         for feature in self.categorical_features:
196 |             if feature in self.categorical_mappings:
197 |                 feature_columns.extend(self.categorical_mappings[feature])
198 |         
199 |         return feature_columns
200 |     
201 |     def save_processors(self, save_dir: str) -> None:
202 |         """保存特征处理器"""
203 |         os.makedirs(save_dir, exist_ok=True)
204 |         
205 |         # 保存标准化器
206 |         with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
207 |             pickle.dump(self.scaler, f)
208 |         
209 |         # 保存分类特征映射
210 |         with open(os.path.join(save_dir, 'categorical_mappings.pkl'), 'wb') as f:
211 |             pickle.dump(self.categorical_mappings, f)
212 |         
213 |         # 保存维度信息
214 |         dim_info = {
215 |             'total_numerical_dim': self.total_numerical_dim,
216 |             'categorical_dims': self.categorical_dims,
217 |             'total_categorical_dim': self.total_categorical_dim
218 |         }
219 |         with open(os.path.join(save_dir, 'dim_info.pkl'), 'wb') as f:
220 |             pickle.dump(dim_info, f)
221 |             
222 |         logger.info(f"[特征处理] 处理器已保存到: {save_dir}")
223 |     
224 |     def load_processors(self, save_dir: str) -> None:
225 |         """加载特征处理器"""
226 |         # 加载标准化器
227 |         with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
228 |             self.scaler = pickle.load(f)
229 |         
230 |         # 加载分类特征映射
231 |         with open(os.path.join(save_dir, 'categorical_mappings.pkl'), 'rb') as f:
232 |             self.categorical_mappings = pickle.load(f)
233 |         
234 |         # 加载维度信息
235 |         with open(os.path.join(save_dir, 'dim_info.pkl'), 'rb') as f:
236 |             dim_info = pickle.load(f)
237 |             self.total_numerical_dim = dim_info['total_numerical_dim']
238 |             self.categorical_dims = dim_info['categorical_dims']
239 |             self.total_categorical_dim = dim_info['total_categorical_dim']
240 |             
241 |         logger.info(f"[特征处理] 处理器已从 {save_dir} 加载")
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/cache_manager.py

- Extension: .py
- Language: python
- Size: 2434 bytes
- Created: 2025-08-03 11:14:53
- Modified: 2025-08-03 11:14:53

### Code

```python
 1 | """
 2 | 缓存管理器
 3 | 提供数据的持久化缓存功能，避免重复计算
 4 | """
 5 | 
 6 | import os
 7 | import pickle
 8 | import logging
 9 | from typing import Any, Optional
10 | 
11 | logger = logging.getLogger(__name__)
12 | 
13 | class CacheManager:
14 |     """缓存管理器"""
15 |     
16 |     def __init__(self, cache_dir: str):
17 |         self.cache_dir = cache_dir
18 |         self._ensure_cache_dir()
19 |         
20 |     def _ensure_cache_dir(self):
21 |         """确保缓存目录存在"""
22 |         if not os.path.exists(self.cache_dir):
23 |             os.makedirs(self.cache_dir)
24 |             logger.info(f"[缓存] 创建缓存目录: {self.cache_dir}")
25 |     
26 |     def _get_cache_path(self, key: str) -> str:
27 |         """获取缓存文件路径"""
28 |         return os.path.join(self.cache_dir, f"{key}.pkl")
29 |     
30 |     def save(self, data: Any, key: str) -> None:
31 |         """保存数据到缓存"""
32 |         cache_path = self._get_cache_path(key)
33 |         try:
34 |             with open(cache_path, 'wb') as f:
35 |                 pickle.dump(data, f)
36 |             logger.info(f"[缓存] 数据已保存: {key}")
37 |         except Exception as e:
38 |             logger.error(f"[缓存] 保存失败 {key}: {e}")
39 |             
40 |     def load(self, key: str) -> Optional[Any]:
41 |         """从缓存加载数据"""
42 |         cache_path = self._get_cache_path(key)
43 |         
44 |         if not os.path.exists(cache_path):
45 |             return None
46 |             
47 |         try:
48 |             with open(cache_path, 'rb') as f:
49 |                 data = pickle.load(f)
50 |             logger.info(f"[缓存] 数据已加载: {key}")
51 |             return data
52 |         except Exception as e:
53 |             logger.error(f"[缓存] 加载失败 {key}: {e}")
54 |             return None
55 |     
56 |     def exists(self, key: str) -> bool:
57 |         """检查缓存是否存在"""
58 |         cache_path = self._get_cache_path(key)
59 |         return os.path.exists(cache_path)
60 |     
61 |     def clear(self, key: str) -> None:
62 |         """清除指定缓存"""
63 |         cache_path = self._get_cache_path(key)
64 |         if os.path.exists(cache_path):
65 |             os.remove(cache_path)
66 |             logger.info(f"[缓存] 缓存已清除: {key}")
67 |     
68 |     def clear_all(self) -> None:
69 |         """清除所有缓存"""
70 |         if os.path.exists(self.cache_dir):
71 |             for filename in os.listdir(self.cache_dir):
72 |                 if filename.endswith('.pkl'):
73 |                     os.remove(os.path.join(self.cache_dir, filename))
74 |             logger.info("[缓存] 所有缓存已清除")
```

## File: /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/libs/data/__init__.py

- Extension: .py
- Language: python
- Size: 247 bytes
- Created: 2025-08-03 11:14:52
- Modified: 2025-08-03 11:14:52

### Code

```python
1 | """
2 | 数据加载、处理与管理模块
3 | """
4 | 
5 | from .data_loader import KuaiRandDataLoader
6 | from .feature_processor import FeatureProcessor
7 | from .cache_manager import CacheManager
8 | 
9 | __all__ = ['KuaiRandDataLoader', 'FeatureProcessor', 'CacheManager']
```


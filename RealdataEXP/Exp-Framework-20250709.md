## Code Architecture
- Main Directory
	- Recommendation System
		- Prediction Module
			- <font color="#ff0000">Initial one-time small data training (for initializing model parameters)</font> (Question: Is it necessary to reinitialize for each experiment?)
			- Training Method 1: Weighted Method
				- Main Model
					- train
					- predict
				- Weight Model
					- train
					- predict
			- Method 2: Data Pooling Method
			- Method 3: Data Segmentation Method
			- Method 4: Snapshot Method
		- Push Module
			- call prediction model
			- metric calculation and sorting
			- call to update pushed dataset, update
	- Dataset Management
		- Pure
			- Find features by video_id
			- Find features by user_id
		- 1K
		- 27K
		- Maintain pushed video library
			- Find real metrics
	- Logging and Dashboard
		- Record push history
		- Record model training history
		- ==Generate WebUI, real-time update (runs on server, accessible via VPN on local computer)==
		- Maintain model training weight saving
	- Execution Module
		- Import other modules
		- Initialize parameters
		- Initialize model (whether to reinitialize model in each loop)
		- Main loop starts
			- Import data, randomly divide into experiment and control groups
			- Simulate and log each experimental method
			- Calculate each metric
	- Result Analysis Module
		- Compare test set errors (prediction errors) of each method
		- Compare actual user metrics of each method
		- Compare GTE of each method
		- …………

## Architecture of Various Prediction Model Training Methods (English Version)
1. Weighted Training **(Method in this paper)**
	- First, initialize classifier $G_{\theta_W}: \mathbb{R}^d \rightarrow [0,1]$ to estimate $E [Z|X_E]$; initialize experiment group model $M_{\theta_T}$ and control group model $M_{\theta_C}$
	- In the $t$-th epoch, include $n_t$ users ($n_t < n$), with $t$ starting from 1
		- Randomly group $n_t$ users (generate $Z$ by Bernoulli distribution with parameter p)
		- Their $X$ as features, $Z$ as labels, optimize $G_{\theta_W}$<font color="#ff0000">(originally optimized</font> $M$<font color="#ff0000">, modified here)</font>
			- Neural Network or SVM or Random Forest
		- According to `user_id` and `Z`, find all eligible `Y` in the dataset
			- Two approaches:
				1. Use all eligible Y (all control/experiment group videos pushed to the user)
				2. Randomly sample $m$ of them
		- Calculate weights:
		$$
			W_{T,i,t} = \frac{G_{\theta_W}(X_{i,t})}{p}, \quad W_{C,i,t} = \frac{1-G_{\theta_W}(X_{i,t})}{1-p}.
		$$
		- $X$ as features, $Y$ as labels, update experiment group model by $W_T \mathcal{D}_E \stackrel{d}{=} \mathcal{D}_T$; update control group model by $W_C \mathcal{D}_E \stackrel{d}{=} \mathcal{D}_C$
		- t=t+1
2. Data Pooling: Use all X and Y at time $t$ for each optimization
3. Snapshot: Fit M in advance, then start experiment, equivalent to fitting all X and Y directly without Test
4. Data Segmentation: For each optimization, experiment group uses only $X$ and $Y$ with $Z=1$; control group uses only $X$ and $Y$ with $Z=0$

## A/B Setting (English Version)
### Type 1
- Use metrics: click-through rate (is_click) and play duration (play_time_ms)
- Different A/B combination parameter $\alpha$

#### Variation

- Use metrics: click-through rate (is_click) and long view rate (long_view)
- Different A/B combination parameter $\alpha$

### Type 2

- A uses metrics: click-through rate (is_click) and play duration (play_time_ms)
- B uses metrics: click-through rate (is_click) and long view rate (long_view)

## 代码架构

- 主目录
	- 推荐系统
		- 预测模块
			- <font color="#ff0000">初始的一次性少量数据训练（作为初始化模型参数）</font>（问题：是否需要每次实验重新初始化）
			- 训练方法 1：数据加权方法
				- 主模型
					- train
					- predict
				- 权重模型
					- train
					- predict
			- 方法 2：数据池化方法
			- 方法 3：数据分割方法
			- 方法 4：快照法
		- 推送模块
			- call 预测模型
			- 指标计算与排序
			- call 更新已推送数据集，更新
	- 数据集管理
		- Pure
			- 跟据 video_id 查找特征
			- 跟据 user_id 查找特征
		- 1K
		- 27K
		- 维护已推送的视频库
			- 查找真实的指标
	- 日志与面板
		- 记录推送历史
		- 记录模型训练历史
		- ==产生 WebUI，实时更新（运行在服务器，可通过 VPN 在本地计算机访问）==
		- 维护模型训练权重保存
	- 执行模块
		- 导入其它模块
		- 初始化参数
		- 初始化模型（要不要每个循环都重新初始化模型）
		- 主循环开始
			- 导入数据、随机分实验组、对照组
			- 分别执行各个被实验的方法的模拟并记录日志
			- 计算各个指标
	- 结果分析模块
		- 对比各方法下的测试集误差（预测误差）
		- 对比各方法的用户实际指标
		- 对比各方法的 GTE
		- …………

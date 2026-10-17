# 动力电池 RUL 滚动预测方案

## 1. 项目目标

本方案基于 BatteryML 和 MATR 锂电池循环数据，构建一个面向动力电池全生命周期管理的 RUL（Remaining Useful Life，剩余使用寿命）预测原型。

当前方案的核心目标是：

1. 使用电池循环过程中的电压、容量等运行信息构造退化特征；
2. 利用连续的历史循环建立滚动预测样本；
3. 在距离寿命终点至少 300 个循环时进行早期 RUL 预测；
4. 使用 XGBoost 完成回归预测；
5. 采用电芯级等权指标评价模型，避免长寿命电芯因窗口数量多而主导结果；
6. 为后续 SOH、异常检测、Agent 和 MPC 等功能扩展提供基础。

本方案不是最终的生产系统，而是一个用于算法验证、特征研究和面试项目展示的 PHM 原型。

---

## 2. 数据集与目录

当前实验使用 MATR 数据集中的电芯循环数据。

项目目录：

```text
D:\Projects\BMS PHM\BatteryML-main
```

主要目录：

```text
data\processed\MATR       预处理后的 MATR 电芯数据
configs\baselines\sklearn\xgb
                            XGBoost 配置文件
batteryml\feature         特征提取代码
batteryml\label           RUL 标签构造代码
batteryml\data             数据集和数据变换代码
batteryml\task.py         训练/测试数据构造流程
workspace                  模型、配置、预测结果和检查点
```

原始数据经过 BatteryML 预处理后，会转换为 `.pkl` 文件。每个文件通常对应一个电芯，例如：

```text
MATR_b1c0.pkl
MATR_b1c1.pkl
MATR_b2c47.pkl
```

`.pkl` 是 Python 的序列化文件格式，用于保存已经组织好的 `BatteryData` 对象，便于后续快速加载。

---

## 3. 总体流程

当前方案的完整数据流如下：

```text
MATR 原始循环数据
        ↓
BatteryML 预处理
        ↓
按电芯划分训练集和测试集
        ↓
提取每个循环的 Q-dV/dQ 类曲线特征
        ↓
构造固定参考差分和相邻变化差分
        ↓
只用训练集拟合 Z-score 和 PCA
        ↓
每个循环降维为 50 维
        ↓
构造长度为 50 的滚动窗口
        ↓
保留预测时 RUL ≥ 300 的样本
        ↓
XGBoost 回归预测 RUL
        ↓
按电芯计算宏平均指标
        ↓
输出预测结果和窗口级对照指标
```

---

## 4. 训练集和测试集划分

当前使用 BatteryML 的 `MATRPrimaryTestTrainTestSplitter`。

重要原则：

> 先按电芯划分训练集和测试集，再在每个电芯内部生成滚动窗口。

不能先把所有窗口混在一起再随机划分，否则同一个电芯的相邻窗口可能同时出现在训练集和测试集，导致严重的数据泄露。

当前测试集电芯编号由 BatteryML 的划分器决定。训练和测试电芯完全分离，模型需要学习跨电芯泛化能力。

---

## 5. 特征构造

### 5.1 Qdlin 特征

BatteryML 中的 `Qdlin` 可以理解为：

```text
放电容量 Q 与电压 V 之间的插值曲线
```

原始不同循环的采样点数量和电压点可能不同，因此 BatteryML 会将曲线插值到统一的电压网格，得到固定长度的特征向量。

它不是严格意义上的 dQ/dV，而是统一电压网格上的放电容量曲线特征。

### 5.2 固定参考差分

参考循环设置为第 8 个循环，当前循环为第 n 个循环：

```text
D_n = X_n - X_8
```

其中：

- `X_8`：第 8 个循环的 Qdlin 特征；
- `X_n`：第 n 个循环的 Qdlin 特征；
- `D_n`：相对于参考循环的累计退化特征。

该特征主要描述电池从参考状态到当前状态的整体变化。

### 5.3 相邻变化差分

为了描述近期退化速度，进一步计算：

```text
V_n = D_n - D_(n-1)
```

由于：

```text
D_n = X_n - X_8
D_(n-1) = X_(n-1) - X_8
```

所以：

```text
V_n = X_n - X_(n-1)
```

这里没有额外引入新的信息，但使用累计差分的形式可以使特征构造逻辑更加统一。

### 5.4 当前采用的组合特征

当前配置为：

```yaml
diff_mode: 'both'
```

即同时输入：

```text
[D_n, V_n]
```

其中：

- `D_n` 表示长期累计退化状态；
- `V_n` 表示相邻循环之间的短期变化速度。

两类特征结合后，模型可以同时观察“已经退化了多少”和“最近退化得有多快”。

对应代码：

```text
batteryml\feature\voltage_capacity_matrix.py
```

---

## 6. 滚动窗口

### 6.1 为什么使用滚动窗口

原始基线通常只使用某一个循环的特征，例如第 98 个循环。这种方式无法直接表达近期退化趋势。

因此当前方案使用长度为 50 的滚动窗口：

```text
第一个样本：循环 1 - 50
第二个样本：循环 2 - 51
第三个样本：循环 3 - 52
...
```

每个样本包含连续 50 个循环的特征。

配置为：

```yaml
rolling_window: 50
```

### 6.2 为什么一个电芯有多个样本

如果某个电芯有 1000 个有效循环，窗口长度为 50，则可以构造：

```text
1000 - 50 + 1 = 951 个窗口
```

因此：

- 每个窗口长度都是 50；
- 不同电芯的窗口总数可能不同；
- 寿命较长的电芯会产生更多窗口。

这也是后续采用电芯级等权评价的主要原因。

---

## 7. RUL 标签构造

### 7.1 EOL 定义

当前使用 SOH 降到额定容量 80% 作为寿命终点：

```yaml
eol_soh: 0.8
```

当放电容量满足：

```text
Q_d ≤ nominal_capacity × 0.8
```

对应循环被认为是 EOL 循环。

### 7.2 RUL 定义

对于第 n 个循环：

```text
RUL_n = EOL_cycle - n
```

### 7.3 预测提前量限制

当前只保留：

```yaml
min_rul_for_prediction: 300
min_rul_limit: 300
```

也就是说，训练和测试样本必须满足：

```text
RUL ≥ 300
```

这表示研究的是至少提前 300 个循环的预测问题。

需要区分：

- 离线实验中可以使用完整寿命数据计算标签和筛选样本；
- 在线部署时并不知道真实 EOL，只能使用当前已有数据进行预测。

因此，`300` 是离线训练/评价的预测提前量约束，不是在线系统输入的未来信息。

对应代码：

```text
batteryml\label\rul.py
```

---

## 8. PCA 降维

### 8.1 为什么需要 PCA

如果每个循环的特征维度约为 2000，50 个循环直接拼接后会形成约 100000 维特征：

```text
50 × 2000 = 100000
```

这会造成：

- 内存占用过高；
- 计算时间变长；
- XGBoost 训练成本增加；
- 可能出现 `not enough memory` 错误。

### 8.2 当前 PCA 流程

当前不是先构造 50 循环窗口再 PCA，而是：

```text
每个循环提取原始特征
        ↓
只使用训练集拟合 Z-score
        ↓
只使用训练集拟合 PCA
        ↓
每个循环降到 50 维
        ↓
再构造 50 循环窗口
```

这样每个窗口的特征维度约为：

```text
50 × 50 = 2500
```

相比 100000 维显著降低了内存压力。

配置为：

```yaml
feature_transformation:
    name: 'SequentialDataTransformation'
    transformations:
        - name: 'ZScoreDataTransformation'
        - name: 'PCADataTransformation'
          n_components: 50
          random_state: 0
          svd_solver: 'randomized'
```

重要原则：

> PCA 只能使用训练集拟合，测试集只能使用训练集已经拟合好的 PCA 参数进行变换。

对应代码：

```text
batteryml\data\transformation\pca.py
batteryml\task.py
```

---

## 9. XGBoost 模型

当前模型配置：

```yaml
model:
    name: 'XGBoostRULPredictor'
```

XGBoost 会把每个滚动窗口转换为一个监督学习样本：

```text
输入：50 个循环的降维特征
标签：窗口末端循环对应的 RUL
输出：预测 RUL
```

模型学习的是：

```text
近期退化轨迹 → 剩余循环数
```

XGBoost 的优点：

- 对中小规模数据比较适合；
- 能处理非线性关系；
- 训练速度通常快于复杂深度模型；
- 对特征尺度和缺失值有一定鲁棒性；
- 适合快速建立可靠基线。

局限：

- 本身不是专门的时序模型；
- 需要人工设计窗口和特征；
- 长期时序依赖能力弱于 LSTM、Transformer 等模型；
- 预测不确定性需要额外建模。

---

## 10. 电芯等权评价

### 10.1 为什么不能只使用窗口等权

如果直接把所有窗口混在一起计算指标，窗口多的长寿命电芯会对结果产生更大影响。

例如：

```text
电芯 A：100 个窗口
电芯 B：1000 个窗口
```

窗口等权评价中，电芯 B 的影响约为电芯 A 的 10 倍。

这不完全符合工程需求，因为工程上通常希望每个电芯都被公平评价。

### 10.2 当前评价方式

当前主指标采用电芯级宏平均：

```text
每个电芯单独计算 RMSE、MAE、MAPE
        ↓
对所有电芯的指标取平均
```

形式上：

```text
Macro-RMSE = mean(RMSE_cell_1, ..., RMSE_cell_N)
```

### 10.3 同时保留窗口级指标

为了与 BatteryML 原始基线比较，预测文件中同时保留窗口级指标：

```text
RMSE       电芯等权 RMSE
MAE        电芯等权 MAE
MAPE       电芯等权 MAPE

window_RMSE
window_MAE
window_MAPE
```

因此：

- 主指标用于工程公平评价；
- `window_` 指标用于和原始基线横向对照。

对应代码：

```text
batteryml\data\databundle.py
batteryml\pipeline.py
batteryml\task.py
```

---

## 11. 当前配置文件

当前完整配置文件为：

```text
configs\baselines\sklearn\xgb\matr_1_rolling_w50_rul300_pca50.yaml
```

主要参数：

```yaml
diff_base: 8
diff_mode: 'both'
min_cycle_index: 8
max_cycle_index: 10000
rolling_window: 50
min_rul_for_prediction: 300
eol_soh: 0.8
min_rul_limit: 300
n_components: 50
```

参数含义：

| 参数 | 含义 |
|---|---|
| `diff_base: 8` | 使用第 8 个循环作为参考循环 |
| `diff_mode: both` | 同时使用累计退化和相邻变化特征 |
| `min_cycle_index: 8` | 从第 8 个循环开始处理 |
| `max_cycle_index: 10000` | 不人为限制实际最大循环数 |
| `rolling_window: 50` | 每个样本包含连续 50 个循环 |
| `min_rul_for_prediction: 300` | 只保留 RUL 至少为 300 的样本 |
| `eol_soh: 0.8` | SOH 降到 80% 视为 EOL |
| `n_components: 50` | PCA 将单循环特征降到 50 维 |

---

## 12. 运行方法

### 12.1 进入项目目录

```bat
cd /d "D:\Projects\BMS PHM\BatteryML-main"
```

### 12.2 运行训练和评估

建议使用新的工作区，避免复用旧的检查点和预测文件：

```bat
batteryml run configs\baselines\sklearn\xgb\matr_1_rolling_w50_rul300_pca50.yaml ^
  --workspace .\workspace\rolling_w50_rul300_pca50_cell_macro ^
  --train ^
  --eval
```

Windows CMD 中的 `^` 表示命令换行。如果写成一行，可以使用：

```bat
batteryml run configs\baselines\sklearn\xgb\matr_1_rolling_w50_rul300_pca50.yaml --workspace .\workspace\rolling_w50_rul300_pca50_cell_macro --train --eval
```

### 12.3 结果文件

工作区中常见文件：

```text
latest.ckpt                         最新检查点
时间戳.ckpt                         某次训练的检查点
config_时间戳.yaml                  本次运行保存的配置
predictions_seed_0_时间戳.pkl       预测结果、指标和数据
```

### 12.4 查看指标

```bat
python -c "import pickle,glob; p=glob.glob(r'workspace\rolling_w50_rul300_pca50_cell_macro\predictions*.pkl')[0]; d=pickle.load(open(p,'rb')); print(d['scores'])"
```

---

## 13. 已完成的实验结果

此前实验得到过以下结果：

| 特征/方案 | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| 原始 XGBoost 基线 | 333.66 | 155.50 | 18.91% |
| 参考循环 5、观测循环 150 | 309.58 | 147.99 | 15.96% |
| 参考循环 5、观测循环 50 | 282.54 | 171.68 | 22.39% |
| 仅相邻循环差分 | 309.19 | 190.19 | 24.40% |
| 累计差分 + 相邻变化 | 240.48 | 140.88 | 16.60% |
| 50 循环滚动窗口 + PCA，窗口级 | 165.23 | 107.71 | 14.55% |
| 50 循环滚动窗口 + PCA，电芯级宏平均 | 94.85 | 73.72 | 14.63% |

注意：这些结果的评价口径并不完全相同，不能简单把所有行当作严格可比的排行榜。后续报告结果时必须同时注明：

- 是否使用滚动窗口；
- 是否限制 RUL ≥ 300；
- 是窗口级指标还是电芯级指标；
- 训练测试划分方式；
- EOL 阈值；
- 是否使用 PCA。

---

## 14. 当前结果的正确解读

当前电芯级宏平均结果为：

```text
RMSE：约 94.85 个循环
MAE ：约 73.72 个循环
MAPE：约 14.63%
```

这说明在当前 MATR Primary 测试划分和提前至少 300 个循环的设定下，模型平均每个电芯的绝对误差约为 74 个循环。

该结果具有较好的研究和面试展示价值，但不能直接宣称已经达到 SOTA，原因包括：

1. 滚动窗口任务与 BatteryML 原始单点基线并不完全相同；
2. 电芯级 RMSE 与窗口级 RMSE 的统计方式不同；
3. MATR 数据集电芯数量有限；
4. 目前还没有进行多随机种子统计；
5. 目前尚未和更多近期方法在完全相同的协议下复现比较。

更稳妥的表述是：

> 在提前至少 300 个循环的滚动 RUL 预测场景下，基于累计退化特征、相邻退化速度特征、PCA 和 XGBoost 的方案取得了约 74 个循环的电芯级 MAE，表现接近公开基准中的优秀结果。

---

## 15. 数据泄露控制

当前方案需要重点保证以下事项：

### 15.1 电芯级划分

同一个电芯的窗口不能同时进入训练集和测试集。

### 15.2 变换只使用训练集拟合

Z-score 和 PCA 都只能在训练集上 `fit`，测试集只能 `transform`。

### 15.3 参考循环不能使用未来信息

参考循环固定为第 8 个循环，不能根据完整寿命事后选择一个最优参考点。

### 15.4 RUL 标签只用于监督训练和离线评价

真实 EOL 可以用于离线构造训练标签，但在线预测时不能把真实 EOL 提供给模型输入。

### 15.5 评价时避免窗口数量偏置

主指标采用电芯级宏平均，避免长寿命电芯因窗口数量多而占据过大权重。

---

## 16. 异常检测扩展方向

RUL 回归模型和异常检测不是同一个任务。

当前可以扩展两条路线。

### 16.1 无监督异常检测

使用电压、温度、容量、内阻等特征训练 Isolation Forest：

```text
正常运行特征 → Isolation Forest → 异常分数
```

优点是不需要大量故障标签，适合早期建立原型。

### 16.2 基于预测残差的异常检测

使用 XGBoost 或 LSTM 预测下一循环的特征，再计算残差：

```text
历史 50 个循环
        ↓
预测第 51 个循环
        ↓
和真实第 51 个循环比较
        ↓
残差超过阈值 → 异常
```

阈值可以根据正常验证集残差的 95% 或 99% 分位数确定。

注意：不能直接用“预测 RUL 和真实 RUL 的误差”做在线异常检测，因为在线时通常不知道真实 RUL。

---

## 17. Agent 和 MCP 扩展方向

后续可以将已有 PHM 模型包装成服务，再通过 Agent 调度。

可能的工具接口包括：

```text
get_battery_latest_data(cell_id)
predict_rul(cell_id)
detect_anomaly(cell_id)
query_alarm_history(cell_id)
generate_battery_report(cell_id)
```

整体架构可以是：

```text
MCP 获取电池数据
        ↓
Agent 调用 RUL/异常检测模型
        ↓
模型输出预测值、异常分数和置信信息
        ↓
Agent 生成解释和处置建议
        ↓
写入运营、风控或告警系统
```

需要明确：

- XGBoost/LSTM 是预测模型；
- MCP 是工具调用协议；
- Agent 是任务理解和流程编排层；
- MCP 不会替代 XGBoost 或 LSTM。

---

## 18. MPC 扩展方向

MPC（Model Predictive Control，模型预测控制）与 XGBoost/LSTM 的定位不同。

```text
XGBoost/LSTM：预测未来状态
MPC：根据预测结果计算控制动作
```

例如：

```text
LSTM 预测未来温度和 SOH
        ↓
MPC 优化充电电流和功率
        ↓
满足温度、电压、电流等约束
```

因此未来可以形成：

```text
PHM 模型：感知和预测
MPC：控制和优化
Agent：调度、解释和业务闭环
```

---

## 19. 下一步实验计划

### 第一阶段：确认评价口径

1. 重新运行带电芯编号的版本；
2. 检查电芯级指标和窗口级指标；
3. 输出每个电芯的 RMSE、MAE、MAPE；
4. 分析误差最大的电芯。

### 第二阶段：稳定性验证

1. 使用多个随机种子训练；
2. 报告均值和标准差；
3. 检查模型是否依赖少数电芯；
4. 对不同 RUL 阶段分别评价。

### 第三阶段：模型对比

比较：

```text
Ridge
Random Forest
XGBoost
LSTM
Transformer 或 TCN
```

必须保证所有模型使用相同的：

- 电芯划分；
- 观测窗口；
- RUL 阈值；
- 特征输入；
- 评价指标。

### 第四阶段：工程化扩展

1. 增加预测不确定性区间；
2. 建立异常检测模块；
3. 增加模型版本和实验记录；
4. 封装 RUL 和异常检测 API；
5. 使用 MCP 暴露模型工具；
6. 结合 Agent 生成电池运维分析报告。

---

## 20. 面试时可以如何介绍

可以用下面的结构介绍项目：

> 我基于 BatteryML 搭建了一个动力电池 RUL 预测原型。首先按照电芯划分训练集和测试集，避免同一电芯的滚动窗口发生数据泄露。特征方面，我以第 8 个循环为参考，同时构造当前循环相对参考循环的累计退化特征，以及相邻循环之间的退化速度特征。随后使用只在训练集拟合的 Z-score 和 PCA，将单循环特征降到 50 维，再构造长度为 50 的滚动窗口，并筛选至少提前 300 个循环的预测样本。最后使用 XGBoost 进行 RUL 回归。由于不同电芯产生的窗口数量不同，我采用电芯级宏平均作为主评价指标，同时保留窗口级指标用于和 BatteryML 官方基线比较。当前模型的电芯级 MAE 约为 74 个循环，MAPE 约为 14.6%。

面试中还应主动说明：

- 该结果是研究原型，不等于真实车辆部署效果；
- 真实部署需要考虑工况变化、传感器噪声、分布外电芯和预测不确定性；
- 当前模型只完成预测，还需要进一步接入异常检测、模型服务和业务闭环。


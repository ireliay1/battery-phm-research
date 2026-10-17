# 技术架构

## 1. 总体数据流

```text
原始 MATR 数据
    ↓ batteryml preprocess
BatteryData .pkl
    ↓ train_test_split
按电芯划分 train/test
    ↓ feature extractor
Qdlin / 差分 / 滚动窗口特征
    ↓ train-only transformation
Z-score + PCA
    ↓ label annotator
SOH 或 RUL 标签
    ↓ model
Ridge / XGBoost / RF / LSTM 等
    ↓ pipeline.evaluate
指标、预测 pkl、checkpoint
```

## 2. 模块职责

| 模块 | 职责 | 关键文件 |
|---|---|---|
| CLI | 解析 `download/preprocess/run` 命令 | `bin/batteryml.py` |
| Pipeline | 加载配置、构建数据、训练、评估和保存结果 | `batteryml/pipeline.py` |
| Task | 组织电芯、标签、特征、滑窗和变换 | `batteryml/task.py` |
| BatteryData | 保存电芯、循环和测试协议数据 | `batteryml/data/battery_data.py` |
| Splitter | 生成训练/测试电芯列表 | `batteryml/train_test_split/MATR_split.py` |
| Feature | 从循环数据提取输入特征 | `batteryml/feature/` |
| Label | 构造 SOH/RUL 监督标签 | `batteryml/label/` |
| Transformation | 标准化、对数变换、PCA | `batteryml/data/transformation/` |
| Model | sklearn 或 PyTorch 预测器 | `batteryml/models/` |
| Result | 保存模型、配置、预测和指标 | `workspace/<实验名>/` |

## 3. 当前 RUL 实验实现

`VoltageCapacityMatrixFeatureExtractor` 当前支持：

- `diff_base`：固定参考循环；
- `diff_mode: reference`：`X_n-X_ref`；
- `diff_mode: previous`：相邻循环差分；
- `diff_mode: both`：累计差分和局部差分拼接；
- `rolling_window`：按电芯构造连续循环窗口；
- `min_rul_for_prediction`：过滤提前量不足的 RUL 样本。

滚动模式先提取单循环序列，再使用训练集单循环特征拟合 Z-score/PCA，最后构造窗口，避免先产生高维原始窗口导致内存溢出。

## 4. 当前 SOH 实验实现

`SOHLabelAnnotator`：

```text
SOH_n = max(discharge_capacity_n) / nominal_capacity
```

非滚动基线在作者的第 98 循环特征上预测一个 SOH；滚动实验在每个窗口末端预测对应 SOH。XGBoost 类名仍为 `XGBoostRULPredictor`，但底层是通用 sklearn 回归器，标签决定任务语义。

## 5. 结果对象

预测文件为 pickle 字典，已确认包含：

```text
prediction  模型输出张量
scores      指标字典
data        DataBundle（含测试特征和标签）
seed        随机种子
```

滚动实验还保存每个样本的电芯分组信息，用于电芯级宏平均。旧结果文件可能没有该元数据，不能直接用于新的电芯分组评价。

## 6. 依赖

依赖由 `requirements.txt` 管理，主要包括 NumPy、Pandas、SciPy、scikit-learn、XGBoost、PyYAML、tqdm、h5py、matplotlib 等。当前运行环境的完整版本尚未归档，后续应补充 `pip freeze` 或最小环境清单。

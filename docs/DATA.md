# 数据说明

## 1. 数据来源

使用 BatteryML 支持的 MATR 数据集。作者 README 记录 MATR 为 LFP/graphite，额定容量约 1.1 Ah，电压范围 2.0–3.6 V，原始数据规模 180 个电芯。

本地已确认：

```text
data/processed/MATR/*.pkl：180 个预处理电芯文件
```

原始下载目录未在当前项目中确认；原始下载和处理命令见作者 `README.md`/`dataprepare.md`。

## 2. BatteryData 主要字段

已从本地对象确认的电芯字段包括：

```text
cell_id
cycle_data
form_factor
anode_material
cathode_material
electrolyte_material
nominal_capacity_in_Ah
depth_of_charge / depth_of_discharge
already_spent_cycles
电压、电流限制
reference / description
charge_protocol / discharge_protocol
```

每个 `cycle_data` 对象包含充放电过程信号及容量信息。当前项目主要使用放电容量和 Qdlin 曲线。

## 3. 预处理

典型命令：

```bat
batteryml preprocess MATR <raw_data_dir> <processed_data_dir>
```

预处理把原始数据转换为统一的 `BatteryData` 结构。`.pkl` 是 Python 序列化文件，不是 MySQL 数据库文件。

## 4. 数据划分

当前主实验使用：

```text
MATRPrimaryTestTrainTestSplitter
```

代码中明确列出 41 个训练电芯和 42 个测试电芯，另有 MATR 数据文件未进入该 Primary 实验划分。划分按电芯完成，之后才生成滚动窗口。

## 5. 标签和特征

### Qdlin

BatteryML 通过放电容量与电压关系生成统一电压网格上的向量特征。其命名可理解为 discharge capacity on a linear voltage grid；它不是严格的 dQ/dV。

### SOH

```text
SOH = 当前循环最大放电容量 / 额定容量
```

### RUL

当前 RUL 以容量下降到额定容量 80% 作为 EOL：

```text
RUL_n = EOL_cycle - n
```

RUL 实验额外过滤 `RUL >= 300`。

## 6. 数据版本与限制

- 数据版本：本地 BatteryML 预处理结果；具体原始下载日期待确认。
- 数据文件未纳入当前 Git（项目没有 `.git` 仓库可用历史）。
- 当前只验证 MATR Primary；MATR Secondary/CLO 等划分未完成本项目复现实验。
- 不同电芯循环数差异大，滚动窗口数量不均衡，因此主评价使用电芯级宏平均，同时保留窗口级指标。

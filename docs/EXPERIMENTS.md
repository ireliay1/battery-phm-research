# 实验索引

## 1. 记录规则

每条实验记录包含：目的、配置、命令、数据划分、随机种子、结果文件、结论和限制。历史结果保留在 `workspace/`，不得覆盖。

当前项目没有 Git 历史，因此“代码版本”以配置文件、工作区时间戳和当前源文件为准。

## 2. 已验证实验总表

### RUL 实验

| 实验/工作区 | 主要设置 | RMSE | MAE | MAPE | 状态 |
|---|---|---:|---:|---:|---|
| `test` / `xgb_matr1` | 作者 XGBoost，参考8、观测98 | 333.6646 | 155.5015 | 18.91% | 已验证 |
| `ref5_obs150` | 参考5、观测150 | 309.5811 | 147.9900 | 15.96% | 已验证 |
| `ref5_obs150` | 参考5、观测50 | 282.5354 | 171.6803 | 22.39% | 已验证，指标不一致 |
| `adjacent_obs98` | 单个相邻差分 | 382.8676 | 227.7189 | 27.62% | 已验证，变差 |
| `adjacent_all_8_98` | 所有相邻差分 | 309.1884 | 190.1919 | 24.40% | 已验证 |
| `both_8_98` | 累计差分+相邻差分 | 240.4806 | 140.8790 | 16.60% | 已验证 |
| `rolling_w50_rul300_pca50_v2` | 50窗口、RUL≥300、PCA50 | 165.2313 | 107.7070 | 14.55% | 已验证，窗口级 |
| `rolling_w50_rul300_pca50_cell_macro` | 同上，电芯级主指标 | 94.8531 | 73.7155 | 14.63% | 已验证，宏平均 |

### SOH 实验

| 实验/工作区 | 主要设置 | RMSE | MAE | MAPE | 状态 |
|---|---|---:|---:|---:|---|
| `soh_xgb_author_baseline` | 作者单循环设置，第98循环，XGBoost | 0.009663 | 0.007932 | 0.81195% | 已验证 |
| `soh_xgb_rolling_pca50` | 50窗口、累计+相邻特征、PCA50、XGBoost | 0.012373 | 0.010066 | 1.08677% | 已验证，全窗口宏平均 |

为公平对齐第98循环，滚动方案在第98循环的结果为：

```text
RMSE = 0.010311
MAE  = 0.007647
MAPE = 0.78284%
```

与作者基线相比：RMSE 变差约 6.70%，MAE 改善约 3.34%，MAPE 改善约 3.34%。因此当前不能声称滚动 SOH 方案整体提升。

## 3. RUL 复现实验命令

```bat
cd /d "D:\Projects\BMS PHM\BatteryML-main"
batteryml run configs\baselines\sklearn\xgb\matr_1.yaml --workspace .\workspace\xgb_matr1 --train --eval
```

改进方案：

```bat
batteryml run configs\baselines\sklearn\xgb\matr_1_rolling_w50_rul300_pca50.yaml ^
  --workspace .\workspace\rolling_w50_rul300_pca50_cell_macro ^
  --train ^
  --eval
```

## 4. SOH 复现实验命令

作者方法基线：

```bat
batteryml run configs\baselines\sklearn\xgb\matr_1_soh_baseline.yaml ^
  --workspace .\workspace\soh_xgb_author_baseline ^
  --train ^
  --eval
```

改进方法：

```bat
batteryml run configs\baselines\sklearn\xgb\matr_1_rolling_soh_pca50.yaml ^
  --workspace .\workspace\soh_xgb_rolling_pca50 ^
  --train ^
  --eval
```

## 5. 结果解释

RUL 的 `94.8531` 是电芯级宏平均，不能直接与作者单点基线的 `333.6646` 做严格同口径比较；窗口级 RUL 结果是 `165.2313`。SOH 的作者基线和滚动方案已经按第98循环做了额外对齐，当前结论是滚动方案只改善 MAE/MAPE，未改善 RMSE。

## 6. 实验卡片

以下记录以当前工作区中实际存在的预测文件为准。所有实验使用 `seed=0`；运行环境完整软件版本尚未归档，见 `docs/ISSUES.md` 的 I011。

### E-RUL-001：作者 XGBoost RUL 基线

- 时间：2026-10-15。
- 目的/假设：复现 BatteryML 作者方法，建立 RUL 比较基准。
- 数据：本地 `data/processed/MATR`，`MATRPrimaryTestTrainTestSplitter`，41 训练电芯/42 测试电芯。
- 配置：`configs/baselines/sklearn/xgb/matr_1.yaml`；参考循环 8，单观测循环 98。
- 命令：见本文件第 3 节。
- 输出：`workspace/test/predictions_seed_0_20261015225705.pkl`。
- 结果：RMSE 333.6646，MAE 155.5015，MAPE 18.91%。
- 结论：作者基线复现成功；不是原创算法结果。

### E-RUL-002：参考循环/观测循环敏感性

- 时间：2026-10-17。
- 目的/假设：改变参考循环和观测循环可能改变退化表征。
- 配置：`matr_1_test5-150.yaml` 等；工作区 `ref5_obs150`。
- 输出：`workspace/ref5_obs150/predictions_seed_0_20261017023613.pkl`、`predictions_seed_0_20261017025330.pkl`。
- 结果：两次结果分别为 RMSE 309.5811/282.5354；MAE 147.9900/171.6803；MAPE 15.96%/22.39%。
- 结论：RMSE 下降不代表整体指标都改善，不能只看单一指标。

### E-RUL-003：差分特征对比

- 时间：2026-10-17。
- 目的/假设：累计参考差分和相邻差分分别表达长期状态与近期变化，组合可能更有效。
- 配置/输出：`adjacent_obs98`、`adjacent_all_8_98`、`both_8_98` 工作区中的 `predictions_seed_0_*.pkl`。
- 结果：单相邻差分 RMSE 382.8676；全相邻差分 309.1884；累计+相邻 240.4806。
- 结论：在当前设置下组合特征优于单一差分，但还需要多 seed 和消融确认。

### E-RUL-004：滚动 RUL 与 PCA

- 时间：2026-10-17。
- 目的/假设：50 循环窗口可以捕获退化轨迹，PCA 可降低高维内存压力。
- 配置：`configs/baselines/sklearn/xgb/matr_1_rolling_w50_rul300_pca50.yaml`。
- 输出：`workspace/rolling_w50_rul300_pca50_cell_macro/predictions_seed_0_20261017223930.pkl`。
- 结果：电芯宏平均 RMSE 94.8531，MAE 73.7155，MAPE 14.63%；窗口级 RMSE 165.2313，MAE 107.7070，MAPE 14.55%。
- 结论：电芯宏平均和窗口级指标必须同时报告，二者不是同一评价口径。

### E-SOH-001：作者方法 SOH 基线

- 时间：2026-10-18。
- 目的/假设：用作者单循环 XGBoost 方法作为 SOH 基线。
- 配置：`configs/baselines/sklearn/xgb/matr_1_soh_baseline.yaml`；第98循环特征，SOH 相对容量标签。
- 输出：`workspace/soh_xgb_author_baseline/predictions_seed_0_20261018013635.pkl`。
- 结果：RMSE 0.0096625，MAE 0.0079321，MAPE 0.81195%。
- 结论：SOH 作者基线运行成功。

### E-SOH-002：滚动窗口 SOH 实验

- 时间：2026-10-18。
- 目的/假设：累计退化+相邻变化、50 循环窗口和 PCA 可以提升 SOH 估计。
- 配置：`configs/baselines/sklearn/xgb/matr_1_rolling_soh_pca50.yaml`。
- 输出：`workspace/soh_xgb_rolling_pca50/predictions_seed_0_20261018013842.pkl`。
- 全窗口结果：电芯宏平均 RMSE 0.0123734，MAE 0.0100658，MAPE 1.08677%。
- 第98循环对齐结果：RMSE 0.0103110，MAE 0.0076471，MAPE 0.78284%。
- 对比：第98循环对齐后 RMSE 比基线差约6.70%，MAE/MAPE 好约3.34%。
- 结论：当前不能声称整体提升；需要进一步定位离群电芯和做消融实验。

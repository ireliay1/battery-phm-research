# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch

from typing import List

from batteryml.builders import FEATURE_EXTRACTORS
from batteryml.data.battery_data import BatteryData
from batteryml.feature.base import BaseFeatureExtractor
from batteryml.feature.severson import get_Qdlin, smooth


@FEATURE_EXTRACTORS.register()
class VoltageCapacityMatrixFeatureExtractor(BaseFeatureExtractor):
    def __init__(self,
                 interp_dim: int = 1000,
                 diff_base: int = 9,
                 diff_mode: str = "reference",
                 cycles_to_keep: List[int] = None,
                 min_cycle_index: int = 0,
                 max_cycle_index: int = 99,
                 rolling_window: int = None,
                 min_rul_for_prediction: float = 300.0,
                 use_precalculated_qdlin: bool = False,
                 smooth: bool = True,
                 cycle_average: int = None):
        self.interp_dim = interp_dim
        self.min_cycle_index = min_cycle_index
        self.max_cycle_index = max_cycle_index
        self.rolling_window = rolling_window
        self.min_rul_for_prediction = min_rul_for_prediction
        self.use_precalculated_qdlin = use_precalculated_qdlin

        assert diff_base <= max_cycle_index, (diff_base, max_cycle_index)
        assert diff_base >= min_cycle_index, (diff_base, min_cycle_index)
        self.diff_base = diff_base
        assert diff_mode in ["reference", "previous", "both"]
        self.diff_mode = diff_mode

        if cycles_to_keep is not None and isinstance(cycles_to_keep, int):
            cycles_to_keep = [cycles_to_keep]
        self.cycles_to_keep = cycles_to_keep

        self.smooth = smooth

        # See https://github.com/petermattia/revisit-severson-et-al/blob/main/revisit-severson-et-al.ipynb noqa
        self.cycle_average = cycle_average
        
    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        feature, _ = self.process_cell_with_indices(cell_data)
        return feature

    def process_cell_with_indices(self, cell_data: BatteryData):
        feature = []
        cycle_indices = []

        # 固定参考循环模式
        diff_base_qdlin = None
        if self.diff_mode in ["reference", "both"]:
            diff_base_qdlin = get_Qdlin(
                cell_data,
                cell_data.cycle_data[self.diff_base],
                self.use_precalculated_qdlin)

            if self.smooth:
                diff_base_qdlin = smooth(diff_base_qdlin)

            if self.cycle_average is not None:
                diff_base_qdlin = diff_base_qdlin[..., ::self.cycle_average]

        # 相邻循环模式使用的上一循环特征
        prev_qdlin = None
        # both 模式使用的上一循环累计差分
        prev_reference_diff = None

        for cycle_index, cycle_data in enumerate(cell_data.cycle_data):
            if cycle_index < self.min_cycle_index:
                continue

            if cycle_index > self.max_cycle_index:
                break

            qdlin = get_Qdlin(
                cell_data,
                cycle_data,
                self.use_precalculated_qdlin)

            if self.smooth:
                qdlin = smooth(qdlin)

            if self.cycle_average is not None:
                qdlin = qdlin[..., ::self.cycle_average]

            if self.diff_mode == "reference":
                # 当前循环 - 固定参考循环
                diff_qdlin = qdlin - diff_base_qdlin

                if self.smooth:
                    diff_qdlin = smooth(diff_qdlin)

                feature_tensor = torch.from_numpy(diff_qdlin)

            else:
                if self.diff_mode == "previous":
                    # 第一个循环没有前一个循环，无法计算相邻差分
                    if prev_qdlin is None:
                        prev_qdlin = qdlin
                        continue

                    # 当前循环 - 上一个循环
                    diff_previous_qdlin = qdlin - prev_qdlin

                    # 更新上一循环特征；必须在筛选 cycles_to_keep 之前更新
                    prev_qdlin = qdlin

                    if self.smooth:
                        diff_previous_qdlin = smooth(diff_previous_qdlin)

                    feature_tensor = torch.from_numpy(diff_previous_qdlin)
                else:
                    # 先计算当前循环相对参考循环的累计差分
                    diff_reference_qdlin = qdlin - diff_base_qdlin

                    # 第一个循环没有上一累计差分，无法计算局部变化
                    if prev_reference_diff is None:
                        prev_reference_diff = diff_reference_qdlin
                        continue

                    # (x_n - x_ref) - (x_(n-1) - x_ref)
                    diff_previous_qdlin = (
                        diff_reference_qdlin - prev_reference_diff)

                    # 更新上一循环累计差分；必须在筛选 cycles_to_keep 之前更新
                    prev_reference_diff = diff_reference_qdlin

                    if self.smooth:
                        diff_reference_qdlin = smooth(diff_reference_qdlin)
                        diff_previous_qdlin = smooth(diff_previous_qdlin)

                    feature_tensor = torch.cat([
                        torch.from_numpy(diff_reference_qdlin),
                        torch.from_numpy(diff_previous_qdlin)
                    ], dim=-1)

            if self.cycles_to_keep is not None \
                    and cycle_index not in self.cycles_to_keep:
                continue

            feature.append(feature_tensor)
            cycle_indices.append(cycle_index)

        feature = torch.stack(feature)

        # 填充 NaN 和无穷值
        feature[torch.isnan(feature) | torch.isinf(feature)] = 0.

        return feature, cycle_indices

# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch

from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData

from .base import BaseLabelAnnotator


@LABEL_ANNOTATORS.register()
class RULLabelAnnotator(BaseLabelAnnotator):
    def __init__(self,
                 eol_soh: float = 0.8,
                 pad_eol: bool = True,
                 min_rul_limit: float = 100.0):
        self.eol_soh = eol_soh
        self.pad_eol = pad_eol
        self.min_rul_limit = min_rul_limit

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        label, found_eol = 1, False
        for cycle in cell_data.cycle_data:
            label += 1
            Qd = max(cycle.discharge_capacity_in_Ah)
            if Qd <= cell_data.nominal_capacity_in_Ah * self.eol_soh:
                found_eol = True
                break

        if not found_eol:
            label = label + 1 if self.pad_eol else float('nan')

        if label <= self.min_rul_limit:
            label = float('nan')

        label = torch.tensor(label)
        return label

    def process_cell_by_cycle(self, cell_data: BatteryData) -> torch.Tensor:
        """Return a causal RUL label for every observed cycle.

        The EOL cycle is used only to create the training target. It is not
        included in the feature input at prediction time.
        """
        eol_cycle_index = None
        for cycle_index, cycle in enumerate(cell_data.cycle_data):
            qd = max(cycle.discharge_capacity_in_Ah)
            if qd <= cell_data.nominal_capacity_in_Ah * self.eol_soh:
                eol_cycle_index = cycle_index
                break

        if eol_cycle_index is None:
            if not self.pad_eol:
                return torch.full(
                    (len(cell_data.cycle_data),), float('nan'))
            eol_cycle_index = len(cell_data.cycle_data)

        labels = torch.tensor([
            eol_cycle_index - cycle_index
            for cycle_index in range(len(cell_data.cycle_data))
        ], dtype=torch.float32)

        labels[labels <= self.min_rul_limit] = float('nan')
        return labels

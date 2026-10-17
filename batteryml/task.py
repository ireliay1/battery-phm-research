# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch

from tqdm import tqdm

from batteryml.builders import (
    FEATURE_EXTRACTORS,
    LABEL_ANNOTATORS,
    TRAIN_TEST_SPLITTERS,
    DATA_TRANSFORMATIONS
)
from batteryml.data import BatteryData, DataBundle
from batteryml.data.transformation.base import BaseDataTransformation


class Task:
    def __init__(self,
                 train_test_splitter: dict,
                 feature_extractor: dict,
                 label_annotator: dict,
                 feature_transformation: BaseDataTransformation = None,
                 label_transformation: BaseDataTransformation = None):
        if isinstance(train_test_splitter, dict):
            train_test_splitter = \
                TRAIN_TEST_SPLITTERS.build(train_test_splitter, 'raise')
        if isinstance(feature_extractor, dict):
            feature_extractor = FEATURE_EXTRACTORS.build(feature_extractor)
        if isinstance(label_annotator, dict):
            label_annotator = LABEL_ANNOTATORS.build(label_annotator)
        if isinstance(feature_transformation, dict):
            feature_transformation = DATA_TRANSFORMATIONS.build(
                feature_transformation)
        if isinstance(label_transformation, dict):
            label_transformation = DATA_TRANSFORMATIONS.build(
                label_transformation)

        self.train_test_splitter = train_test_splitter
        self.feature_extractor = feature_extractor
        self.label_annotator = label_annotator
        self.feature_transformation = feature_transformation
        self.label_transformation = label_transformation

    def _extract_rolling_sequences(self, cells):
        sequences = []

        for cell in tqdm(cells, desc='Extracting rolling sequences'):
            sequence, cycle_indices = \
                self.feature_extractor.process_cell_with_indices(cell)
            label_by_cycle = self.label_annotator.process_cell_by_cycle(cell)
            sequences.append((cell.cell_id, sequence, cycle_indices, label_by_cycle))

        return sequences

    def _build_rolling_dataset(self, sequences):
        window = self.feature_extractor.rolling_window
        min_rul = getattr(self.feature_extractor,
                          'min_rul_for_prediction', None)
        features = []
        labels = []

        group_ids = []
        for cell_id, sequence, cycle_indices, label_by_cycle in tqdm(
                sequences, desc='Building rolling samples'):
            if len(sequence) < window:
                continue

            for end_index in range(window - 1, len(sequence)):
                cycle_index = cycle_indices[end_index]
                label = label_by_cycle[cycle_index]

                if not torch.isfinite(label):
                    continue
                if min_rul is not None and label < min_rul:
                    continue

                features.append(sequence[
                    end_index - window + 1:end_index + 1
                ])
                labels.append(label)
                group_ids.append(cell_id)

        if not features:
            raise RuntimeError(
                'No rolling samples were generated. Check the window size '
                'and minimum RUL setting.')

        return torch.stack(features), torch.stack(labels), group_ids

    def build(self) -> DataBundle:
        # Loading data
        train_list, test_list = self.train_test_splitter.split()
        pbar = tqdm(train_list, desc='Reading train data')
        train_cells = [BatteryData.load(path) for path in pbar]
        pbar = tqdm(test_list, desc='Reading test data')
        test_cells = [BatteryData.load(path) for path in pbar]

        self.train_cells = train_cells
        self.test_cells = test_cells

        if getattr(self.feature_extractor, 'rolling_window', None) is not None:
            # The cell-level train/test split happens before windows are
            # generated, preventing windows from the same cell crossing the
            # train/test boundary.
            train_sequences = self._extract_rolling_sequences(train_cells)
            test_sequences = self._extract_rolling_sequences(test_cells)

            # Fit feature transformations on individual train-cycle features
            # before constructing windows. This avoids materializing huge raw
            # 50-cycle windows before PCA can reduce them.
            if self.feature_transformation is not None:
                train_cycle_features = torch.cat([
                    sequence for _, sequence, _, _ in train_sequences
                ], dim=0)
                self.feature_transformation.fit(train_cycle_features)
                train_sequences = [
                    (cell_id, self.feature_transformation.transform(sequence),
                     cycle_indices, rul_by_cycle)
                    for cell_id, sequence, cycle_indices, rul_by_cycle
                    in train_sequences
                ]
                test_sequences = [
                    (cell_id, self.feature_transformation.transform(sequence),
                     cycle_indices, rul_by_cycle)
                    for cell_id, sequence, cycle_indices, rul_by_cycle
                    in test_sequences
                ]

            train_features, train_labels, train_group_ids = \
                self._build_rolling_dataset(train_sequences)
            test_features, test_labels, test_group_ids = \
                self._build_rolling_dataset(test_sequences)

            # Feature transformations were already fitted and applied at the
            # individual-cycle level. DataBundle only needs to transform the
            # labels for rolling datasets.
            feature_transformation = None
        else:
            # Extracting features
            train_features = self.feature_extractor(train_cells)
            test_features = self.feature_extractor(test_cells)
            train_labels = self.label_annotator(train_cells)
            test_labels = self.label_annotator(test_cells)

            # Omit NaN label cells
            train_mask = ~torch.isnan(train_labels)
            test_mask = ~torch.isnan(test_labels)
            train_features = train_features[train_mask]
            test_features = test_features[test_mask]
            train_labels = train_labels[train_mask]
            test_labels = test_labels[test_mask]

            feature_transformation = self.feature_transformation
            train_group_ids = None
            test_group_ids = None

        dataset = DataBundle(
            train_features, train_labels, test_features, test_labels,
            feature_transformation=feature_transformation,
            label_transformation=self.label_transformation,
            train_group_ids=train_group_ids,
            test_group_ids=test_group_ids
        )

        return dataset

    def get_raw_data(self):
        return self.train_cells, self.test_cells

"""Isolation Forest anomaly detection with a MAD-derived threshold."""

from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from batteryml.data.battery_data import BatteryData
from batteryml.feature.voltage_capacity_matrix import (
    VoltageCapacityMatrixFeatureExtractor,
)
from batteryml.train_test_split.MATR_split import (
    MATRPrimaryTestTrainTestSplitter,
)


def load_features(paths, extractor, start_cycle, end_cycle):
    rows, metadata = [], []
    for path in paths:
        cell = BatteryData.load(path)
        features, cycle_indices = extractor.process_cell_with_indices(cell)
        features = features.detach().cpu().numpy().astype(np.float32)
        for row, cycle in zip(features, cycle_indices):
            if start_cycle <= cycle <= end_cycle:
                rows.append(row)
                metadata.append((cell.cell_id, int(cycle)))
    if not rows:
        raise RuntimeError("No features were extracted for the requested cycle range")
    return np.asarray(rows), metadata


def mad_threshold(scores, k):
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust_std = 1.4826 * mad
    threshold = median + k * robust_std
    return threshold, median, mad, robust_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/MATR")
    parser.add_argument("--output", default="workspace/anomaly_iforest_mad")
    parser.add_argument("--k", type=float, default=3.5)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--synthetic-shift", type=float, default=5.0)
    parser.add_argument("--synthetic-dimensions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    train_paths, test_paths = MATRPrimaryTestTrainTestSplitter(
        str(data_path)).split()
    extractor = VoltageCapacityMatrixFeatureExtractor(
        diff_base=8,
        diff_mode="both",
        min_cycle_index=8,
        max_cycle_index=10000,
        use_precalculated_qdlin=True,
        aggregation_bins=20,
    )

    # Early training cycles are treated as the normal reference population.
    x_train, _ = load_features(train_paths, extractor, 8, 98)
    # This slice is used only to estimate the normal-score threshold.
    x_val, val_meta = load_features(train_paths, extractor, 99, 198)
    x_test, test_meta = load_features(test_paths, extractor, 8, 10000)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_train)

    # Larger values consistently mean "more anomalous".
    val_scores = -model.decision_function(x_val)
    test_scores = -model.decision_function(x_test)
    threshold, median, mad, robust_std = mad_threshold(val_scores, args.k)
    real_alarm = test_scores > threshold

    # Synthetic benchmark only: perturb a known subset of test feature vectors.
    rng = np.random.default_rng(args.seed)
    n_anomaly = max(1, int(round(len(x_test) * 0.10)))
    anomaly_indices = rng.choice(len(x_test), n_anomaly, replace=False)
    x_synthetic = x_test.copy()
    for row_index in anomaly_indices:
        feature_indices = rng.choice(
            x_test.shape[1],
            size=min(args.synthetic_dimensions, x_test.shape[1]),
            replace=False,
        )
        x_synthetic[row_index, feature_indices] += args.synthetic_shift
    synthetic_scores = -model.decision_function(x_synthetic)
    synthetic_pred = synthetic_scores > threshold
    synthetic_true = np.zeros(len(x_test), dtype=bool)
    synthetic_true[anomaly_indices] = True

    result = {
        "dataset": "MATR",
        "split": "MATRPrimaryTestTrainTestSplitter",
        "feature": "Qdlin reference+adjacent difference, 20-bin mean+std",
        "train_cycles": "8-98",
        "validation_cycles": "99-198 (assumed normal)",
        "threshold_method": "median + k * 1.4826 * MAD",
        "k": args.k,
        "threshold": threshold,
        "validation_score_median": median,
        "validation_MAD": mad,
        "validation_robust_std": robust_std,
        "real_test_samples": len(test_scores),
        "real_test_alarm_count": int(real_alarm.sum()),
        "real_test_alarm_rate": float(real_alarm.mean()),
        "synthetic_anomaly_fraction": float(synthetic_true.mean()),
        "synthetic_shift": args.synthetic_shift,
        "synthetic_dimensions": args.synthetic_dimensions,
        "synthetic_precision": float(
            precision_score(synthetic_true, synthetic_pred, zero_division=0)),
        "synthetic_recall": float(
            recall_score(synthetic_true, synthetic_pred, zero_division=0)),
        "synthetic_f1": float(
            f1_score(synthetic_true, synthetic_pred, zero_division=0)),
        "seed": args.seed,
        "notes": [
            "Real MATR test cells have no cycle-level fault labels; alarm rate is not a true FPR.",
            "Synthetic F1 is only a pipeline sanity check, not a real-fault benchmark.",
        ],
    }
    (output_path / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "metadata.json").write_text(
        json.dumps({"validation_rows": val_meta, "test_rows": test_meta},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

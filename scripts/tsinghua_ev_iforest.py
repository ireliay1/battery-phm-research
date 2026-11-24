"""Vehicle-level Isolation Forest benchmark on the Tsinghua EV dataset.

The released labels are vehicle-level. Therefore all snippets from one vehicle
are aggregated before the final metrics are computed. Only normal training
vehicles are used to fit the model and estimate the MAD threshold.
"""

from pathlib import Path
import argparse
import csv
import io
import json
import tarfile

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


COLUMNS = [
    "volt", "current", "soc", "max_single_volt", "min_single_volt",
    "max_temp", "min_temp", "timestamp",
]


def linear_slope(signal):
    signal = np.asarray(signal, dtype=float)
    x = np.arange(len(signal), dtype=float)
    mask = np.isfinite(signal)
    if mask.sum() <= 1:
        return 0.0
    xv = x[mask]
    yv = signal[mask]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denominator = np.dot(xv, xv)
    return float(np.dot(xv, yv) / denominator) if denominator else 0.0


def read_label_table(tar, prefix):
    member = tar.getmember(prefix)
    rows = csv.DictReader(io.TextIOWrapper(tar.extractfile(member), encoding="utf-8"))
    return {int(row["car"]): int(row["label"]) for row in rows}


def read_label_file(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {int(row["car"]): int(row["label"]) for row in rows}


def feature_vector(array):
    """Create fixed-length, interpretable features for one charging snippet."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(COLUMNS):
        raise ValueError(f"Unexpected snippet shape: {array.shape}")

    # Timestamp is an index, not a physical battery feature.
    signal = array[:, :7]
    finite = np.where(np.isfinite(signal), signal, np.nan)
    mean = np.nanmean(finite, axis=0)
    std = np.nanstd(finite, axis=0)
    minimum = np.nanmin(finite, axis=0)
    maximum = np.nanmax(finite, axis=0)
    first = finite[0]
    last = finite[-1]
    delta = last - first
    slope = np.zeros(signal.shape[1])
    for column in range(signal.shape[1]):
        slope[column] = linear_slope(finite[:, column])
    return np.concatenate([mean, std, minimum, maximum, first, last, delta, slope])


def response_input(array):
    soc = np.asarray(array[:, 2], dtype=float)
    current = np.asarray(array[:, 1], dtype=float)
    return np.column_stack([
        soc, current, soc ** 2, current ** 2, soc * current,
    ])


def read_brand_labels(tar_path, brand):
    root = f"battery_brand{brand}/"
    with tarfile.open(tar_path, mode="r:gz") as tar:
        return (
            read_label_table(tar, root + "label/train_label.csv"),
            read_label_table(tar, root + "label/test_label.csv"),
        )


def read_directory_labels(data_root, brand):
    root = Path(data_root) / f"battery_brand{brand}"
    return (
        read_label_file(root / "label" / "train_label.csv"),
        read_label_file(root / "label" / "test_label.csv"),
    )


def fit_response_models(tar_path, brand, fit_cars, max_points=200000):
    """Fit normal SOC/current -> voltage/temperature response models."""
    prefix = f"battery_brand{brand}/train/"
    x_parts, y_parts = [], []
    with tarfile.open(tar_path, mode="r:gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.startswith(prefix)
                    and member.name.endswith(".pkl")):
                continue
            sequence, metadata = torch.load(
                io.BytesIO(tar.extractfile(member).read()),
                map_location="cpu", weights_only=False)
            if int(metadata["car"]) not in fit_cars:
                continue
            array = np.asarray(sequence, dtype=float)
            # Two regularly spaced points per snippet are enough to learn the
            # coarse normal response while keeping the training set bounded.
            indices = np.linspace(0, len(array) - 1, 2, dtype=int)
            x_parts.append(response_input(array[indices]))
            y_parts.append(array[indices][:, [0, 5, 6]])
            if sum(len(part) for part in x_parts) >= max_points:
                break
    x = np.concatenate(x_parts, axis=0)[:max_points]
    y = np.concatenate(y_parts, axis=0)[:max_points]
    models = []
    for target in range(y.shape[1]):
        model = Ridge(alpha=1.0)
        model.fit(x, y[:, target])
        models.append(model)
    return models


def fit_response_models_directory(data_root, brand, fit_cars, max_points=200000):
    """Fit the same normal-response models from an extracted directory."""
    root = Path(data_root) / f"battery_brand{brand}" / "train"
    x_parts, y_parts = [], []
    for path in sorted(root.glob("*.pkl")):
        sequence, metadata = torch.load(
            path, map_location="cpu", weights_only=False)
        if int(metadata["car"]) not in fit_cars:
            continue
        array = np.asarray(sequence, dtype=float)
        indices = np.linspace(0, len(array) - 1, 2, dtype=int)
        x_parts.append(response_input(array[indices]))
        y_parts.append(array[indices][:, [0, 5, 6]])
        if sum(len(part) for part in x_parts) >= max_points:
            break
    x = np.concatenate(x_parts, axis=0)[:max_points]
    y = np.concatenate(y_parts, axis=0)[:max_points]
    models = []
    for target in range(y.shape[1]):
        model = Ridge(alpha=1.0)
        model.fit(x, y[:, target])
        models.append(model)
    return models


def dynamic_features(array, response_models):
    """Add voltage consistency, temperature-rate and response residual features."""
    array = np.asarray(array, dtype=float)
    x = response_input(array)
    residuals = []
    for target, model in zip([0, 5, 6], response_models):
        residuals.append(array[:, target] - model.predict(x))

    extra = []
    voltage_spread = array[:, 3] - array[:, 4]
    temperature_mean = (array[:, 5] + array[:, 6]) / 2.0
    signals = [voltage_spread, np.gradient(temperature_mean)] + residuals
    for signal in signals:
        signal = np.asarray(signal, dtype=float)
        extra.extend([
            np.nanmean(signal),
            np.nanstd(signal),
            np.nanmax(np.abs(signal)),
            np.nanmean(signal[-1:] - signal[:1]),
            linear_slope(signal),
        ])
    return np.concatenate([feature_vector(array), np.asarray(extra)])


def collect_brand(tar_path, brand, feature_mode="baseline", response_models=None):
    root = f"battery_brand{brand}/"
    train_prefix = root + "train/"
    test_prefix = root + "test/"
    train_rows, test_rows = [], []
    with tarfile.open(tar_path, mode="r:gz") as tar:
        train_labels = read_label_table(tar, root + "label/train_label.csv")
        test_labels = read_label_table(tar, root + "label/test_label.csv")
        members = [m for m in tar if m.isfile() and m.name.endswith(".pkl")]
        for member in members:
            if not (member.name.startswith(train_prefix)
                    or member.name.startswith(test_prefix)):
                continue
            payload = tar.extractfile(member).read()
            sequence, metadata = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=False)
            car = int(metadata["car"])
            row = {
                "feature": (dynamic_features(sequence, response_models)
                            if feature_mode == "dynamic"
                            else feature_vector(sequence)),
                "car": car,
                "label": (train_labels if member.name.startswith(train_prefix)
                           else test_labels)[car],
                "split": "train" if member.name.startswith(train_prefix) else "test",
            }
            (train_rows if row["split"] == "train" else test_rows).append(row)
    return train_rows, test_rows


def collect_directory(data_root, brand, feature_mode="baseline", response_models=None,
                      max_snippets_per_car=None, samples_per_car=None, seed=0):
    """Read an extracted brand directory without scanning a compressed tar."""
    root = Path(data_root) / f"battery_brand{brand}"
    train_labels = read_label_file(root / "label" / "train_label.csv")
    test_labels = read_label_file(root / "label" / "test_label.csv")
    train_rows, test_rows = [], []
    for split, labels, target in (
        ("train", train_labels, train_rows), ("test", test_labels, test_rows)
    ):
        paths_by_car = {}
        for path in sorted((root / split).glob("*.pkl")):
            sequence, metadata = torch.load(path, map_location="cpu", weights_only=False)
            car = int(metadata["car"])
            paths_by_car.setdefault(car, []).append(path)
        rng = np.random.default_rng(seed)
        for car in sorted(paths_by_car):
            paths = paths_by_car[car]
            limit = samples_per_car or max_snippets_per_car
            if limit is not None and len(paths) > limit:
                selected = rng.choice(len(paths), size=limit, replace=False)
                paths = [paths[int(index)] for index in sorted(selected)]
            for path in paths:
                sequence, metadata = torch.load(
                    path, map_location="cpu", weights_only=False)
                feature = (dynamic_features(sequence, response_models)
                           if feature_mode == "dynamic" else feature_vector(sequence))
                target.append({"feature": feature, "car": car,
                               "label": labels[car], "split": split})
    return train_rows, test_rows


def quantile_threshold(scores, quantile):
    return float(np.quantile(scores, quantile))


def aggregate_vehicle_scores(rows, scores, top_fraction):
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row["car"], []).append(float(score))
    result = {}
    for car, values in grouped.items():
        values = np.sort(np.asarray(values))
        count = max(1, int(np.ceil(len(values) * top_fraction)))
        result[car] = float(values[-count:].mean())
    return result


def aggregate_vehicle_features(rows):
    """Convert snippet features to one robust distribution vector per vehicle."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["car"], []).append(row["feature"])
    result = {}
    for car, values in grouped.items():
        matrix = np.asarray(values, dtype=np.float32)
        # Vehicle-level distribution shape is often more informative than the
        # average anomaly score of individual snippets.
        result[car] = np.concatenate([
            np.nanmean(matrix, axis=0),
            np.nanstd(matrix, axis=0),
            np.nanquantile(matrix, 0.10, axis=0),
            np.nanquantile(matrix, 0.50, axis=0),
            np.nanquantile(matrix, 0.90, axis=0),
        ])
    return result


def evaluate(scores, labels, threshold):
    prediction = scores > threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_predictions": int(prediction.sum()),
        "samples": int(len(labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data-root", default=None,
                        help="Optional extracted root containing battery_brandN")
    parser.add_argument("--max-snippets-per-car", type=int, default=None,
                        help="Optional deterministic cap for an I/O pilot")
    parser.add_argument("--samples-per-car", type=int, default=None,
                        help="Randomly sample this many snippets per vehicle")
    parser.add_argument("--brand", type=int, default=1)
    parser.add_argument("--output", default="workspace/tsinghua_ev_iforest_brand1")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--model-level", choices=["snippet", "vehicle"],
                        default="snippet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-mode", choices=["baseline", "dynamic"],
                        default="baseline")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.data_root:
        train_labels, test_labels = read_directory_labels(args.data_root, args.brand)
    else:
        train_labels, test_labels = read_brand_labels(args.archive, args.brand)
    normal_cars = [car for car, label in train_labels.items() if label == 0]
    if not normal_cars:
        raise RuntimeError("No normal training vehicles found")

    # Split normal vehicles before fitting, so threshold calibration is independent.
    rng = np.random.default_rng(args.seed)
    normal_cars = sorted(normal_cars)
    rng.shuffle(normal_cars)
    fit_count = max(1, int(len(normal_cars) * 0.8))
    fit_cars = set(normal_cars[:fit_count])
    calibration_cars = set(normal_cars[fit_count:])

    response_models = None
    if args.feature_mode == "dynamic":
        response_models = (fit_response_models_directory(
            args.data_root, args.brand, fit_cars)
            if args.data_root else fit_response_models(
                args.archive, args.brand, fit_cars))
    if args.data_root:
        train_rows, test_rows = collect_directory(
            args.data_root, args.brand, feature_mode=args.feature_mode,
            response_models=response_models,
            max_snippets_per_car=args.max_snippets_per_car,
            samples_per_car=args.samples_per_car, seed=args.seed)
    else:
        train_rows, test_rows = collect_brand(
            args.archive, args.brand, feature_mode=args.feature_mode,
            response_models=response_models)
    train_normal_all = [row for row in train_rows if row["label"] == 0]
    train_fit = [row for row in train_normal_all if row["car"] in fit_cars]
    train_calibration = [
        row for row in train_normal_all if row["car"] in calibration_cars]

    if args.model_level == "vehicle":
        fit_features = aggregate_vehicle_features(train_fit)
        calibration_features = aggregate_vehicle_features(train_calibration)
        test_features = aggregate_vehicle_features(test_rows)
        fit_cars_order = sorted(fit_features)
        calibration_cars_order = sorted(calibration_features)
        test_cars = sorted(test_features)
        x_fit = np.asarray([fit_features[car] for car in fit_cars_order])
        x_calibration = np.asarray([
            calibration_features[car] for car in calibration_cars_order])
        x_test = np.asarray([test_features[car] for car in test_cars])
    else:
        x_fit = np.asarray([row["feature"] for row in train_fit])
        x_calibration = np.asarray([row["feature"] for row in train_calibration])
        x_test = np.asarray([row["feature"] for row in test_rows])
    scaler = StandardScaler().fit(x_fit)
    x_fit = scaler.transform(x_fit)
    x_calibration = scaler.transform(x_calibration)
    x_test = scaler.transform(x_test)

    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_fit)
    calibration_scores = -model.decision_function(x_calibration)
    test_scores = -model.decision_function(x_test)

    if args.model_level == "vehicle":
        calibration_vehicle = dict(zip(calibration_cars_order, calibration_scores))
        test_vehicle = dict(zip(test_cars, test_scores))
    else:
        calibration_vehicle = aggregate_vehicle_scores(
            train_calibration, calibration_scores, args.top_fraction)
        test_vehicle = aggregate_vehicle_scores(test_rows, test_scores, args.top_fraction)
    calibration_vehicle_scores = np.asarray(list(calibration_vehicle.values()))
    vehicle_threshold = quantile_threshold(
        calibration_vehicle_scores, args.quantile)
    test_cars = sorted(test_vehicle)
    test_vehicle_scores = np.asarray([test_vehicle[car] for car in test_cars])
    test_labels = np.asarray([
        next(row["label"] for row in test_rows if row["car"] == car)
        for car in test_cars
    ])

    result = {
        "dataset": "Tsinghua EV",
        "brand": args.brand,
        "train_snippets": len(train_rows),
        "normal_train_snippets": len(train_normal_all),
        "fit_normal_vehicles": len(fit_cars),
        "calibration_normal_vehicles": len(calibration_cars),
        "fit_normal_snippets": len(train_fit),
        "calibration_normal_snippets": len(train_calibration),
        "test_snippets": len(test_rows),
        "normal_train_vehicles": len({row["car"] for row in train_normal_all}),
        "test_vehicles": len(test_cars),
        "test_abnormal_vehicles": int(test_labels.sum()),
        "max_snippets_per_car": args.max_snippets_per_car,
        "samples_per_car": args.samples_per_car,
        "feature_dimension": int(x_fit.shape[1]),
        "feature_definition": (
            "baseline statistics plus voltage spread, temperature rate, "
            "and SOC/current response residual statistics"
            if args.feature_mode == "dynamic" else
            "7 channels x (mean,std,min,max,first,last,delta,slope), timestamp excluded"),
        "feature_mode": args.feature_mode,
        "n_estimators": args.n_estimators,
        "threshold_method": "normal calibration vehicle-score quantile",
        "quantile": args.quantile,
        "top_fraction": args.top_fraction,
        "model_level": args.model_level,
        "vehicle_threshold": float(vehicle_threshold),
        "vehicle_metrics": evaluate(test_vehicle_scores, test_labels, vehicle_threshold),
        "seed": args.seed,
        "notes": [
            "Labels are vehicle-level; all snippets from a vehicle stay in one split.",
            "Only normal fitting vehicles fit the scaler and Isolation Forest; separate normal calibration vehicles set the quantile threshold.",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

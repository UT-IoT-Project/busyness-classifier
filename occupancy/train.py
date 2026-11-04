"""Train occupancy classification or regression models.

Examples from the repository root:
    python occupancy\train.py --api --start "2026-10-01 00:00" --end "2026-11-04 00:00"
    python occupancy\train.py --api --rooms 1 2 --start "2026-10-01 00:00" --end "2026-11-04 00:00"
    python occupancy\train.py --simulated
    python occupancy\train.py --api --regression --start "2026-10-01 00:00" --end "2026-11-04 00:00"

API training uses a chronological 80/20 train/test split for each room. This
keeps later readings out of training and makes overfitting visible.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occupancy.api_client import ApiSource, format_time
from occupancy.cleaning import clean
from occupancy.features import LABEL_COLUMN, build_windows
from occupancy.integration import merge_all

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0, 1, 2, 3]
# Use 12-hour chunks so local daylight-saving transitions cannot exceed the
# API's 24-hour duration check.
MAX_API_RANGE = timedelta(hours=12)
TEST_SIZE = 0.2


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_range(source, start, end):
    """Fetch an inclusive range in API-sized chunks."""
    merged = {}
    chunks_with_data = 0
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + MAX_API_RANGE, end)
        print(f"  retrieving {short_time(chunk_start)} -> {short_time(chunk_end)}")
        data = source.fetch(chunk_start, chunk_end)
        if data:
            chunks_with_data += 1
            for sensor, readings in data.items():
                merged.setdefault(sensor, []).extend(readings)
        else:
            print("    no readings in this chunk; continuing")
        chunk_start = chunk_end + timedelta(milliseconds=1)
    if not merged:
        return None
    print(f"  received data in {chunks_with_data} chunk(s)")
    return merged


def short_time(moment):
    """Format timestamps for progress output without seconds or year."""
    return moment.strftime("%d %b %H:%M")


def load_sessions(args, config):
    """Load labelled sessions from simulation, files, or the real API."""
    if args.simulated:
        with open(ROOT / "data" / "simulated" / "readings.json", encoding="utf-8") as f:
            return [("simulated", json.load(f)["data"])]

    if args.api:
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
        if end <= start:
            raise SystemExit("--end must be later than --start")
        sessions = []
        for room in args.rooms:
            room_config = dict(config)
            room_config["api"] = dict(config["api"], room=str(room))
            print(f"\nRetrieving designlab room {room}: "
                  f"{short_time(start)} -> {short_time(end)}")
            data = fetch_range(ApiSource(room_config), start, end)
            if data is None:
                print(f"  room {room}: no readings in the requested period; skipping")
                continue
            sessions.append((f"designlab-room-{room}", data))
        return sessions

    sessions = []
    for path in sorted((ROOT / "data" / "raw").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            sessions.append((path.stem, json.load(f)["data"]))
    return sessions


def make_windows(data, config):
    """Clean one API response, integrate it, and create labelled windows."""
    tables, pir, labels = clean(data, config["cleaning"])
    df = merge_all(tables, pir, labels)
    window_config = dict(config["windows"])
    window_config["min_sensor_fraction"] = 1 / len(config["windows"]["sensors"])
    return build_windows(
        df,
        config["windows"]["sensors"],
        window_config,
        config["cleaning"]["interval_seconds"],
        label_column=config["cleaning"]["label"]["sensor"],
    )


def new_model(regression):
    model_type = RandomForestRegressor if regression else RandomForestClassifier
    return model_type(n_estimators=200, random_state=42,
                      **({} if regression else {"class_weight": "balanced"}))


def split_windows(windows, test_size):
    """Split each session chronologically, preventing future leakage."""
    train_parts, test_parts = [], []
    for session, group in windows.groupby("session", sort=False):
        group = group.sort_index()
        split = int(len(group) * (1 - test_size))
        if split < 1 or split >= len(group):
            raise SystemExit(f"Session {session} needs at least 2 windows")
        train_parts.append(group.iloc[:split])
        test_parts.append(group.iloc[split:])
    return pd.concat(train_parts), pd.concat(test_parts)


def plot_raw_data(data, room, output):
    """Save readable raw sensor time-series plots before cleaning."""
    scalar_sensors = [name for name, readings in data.items()
                      if name != "pirs" and readings][:8]
    if not scalar_sensors:
        return
    columns = 2
    rows = (len(scalar_sensors) + 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3 * rows), squeeze=False)
    for axis, sensor in zip(axes.flat, scalar_sensors):
        readings = data[sensor]
        times = [datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
                 for item in readings]
        values = [item["value"] for item in readings]
        axis.plot(times, values, linewidth=0.7)
        axis.set_title(sensor)
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    for axis in axes.flat[len(scalar_sensors):]:
        axis.set_visible(False)
    figure.suptitle(f"Raw sensor data - Designlab room {room}")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_classification_evaluation(
        train_score, test_score, y_test, test_pred, output):
    """Save accuracy, confusion, and green/red prediction plots."""
    matrix = confusion_matrix(y_test, test_pred, labels=LEVELS)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5),
                                gridspec_kw={"width_ratios": [1, 1, 1.4]})

    axes[0].bar(["train", "test"], [train_score, test_score],
                color=["#4c78a8", "#f58518"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Accuracy\n(overfit gap: {train_score - test_score:+.3f})")
    for index, value in enumerate([train_score, test_score]):
        axes[0].text(index, value + 0.02, f"{value:.3f}", ha="center")
    axes[0].grid(axis="y", alpha=0.25)

    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_title("Test confusion matrix")
    axes[1].set_xlabel("Predicted level")
    axes[1].set_ylabel("Recorded level")
    axes[1].set_xticks(range(len(LEVELS)), LEVELS)
    axes[1].set_yticks(range(len(LEVELS)), LEVELS)
    for row in range(len(LEVELS)):
        for column in range(len(LEVELS)):
            axes[1].text(column, row, matrix[row, column],
                         ha="center", va="center",
                         color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    samples = np.arange(len(y_test))
    correct = np.asarray(y_test) == np.asarray(test_pred)
    axes[2].scatter(samples[correct], np.asarray(y_test)[correct],
                    color="#2ca02c", label="correct", s=45)
    axes[2].scatter(samples[~correct], np.asarray(y_test)[~correct],
                    color="#d62728", label="wrong", s=55, marker="x")
    axes[2].scatter(samples, test_pred, color="#222222", alpha=0.35,
                    label="guess", s=18)
    axes[2].set_title("Test guesses vs recorded labels")
    axes[2].set_xlabel("Test slot")
    axes[2].set_ylabel("Occupancy level")
    axes[2].set_yticks(LEVELS)
    axes[2].set_ylim(-0.4, 3.4)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.suptitle(f"Classification evaluation ({len(y_test)} held-out slots)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_regression_evaluation(train_score, test_score, output):
    """Save a train/test comparison plot for regression."""
    labels = ["train", "test"]
    values = [train_score, test_score]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#4c78a8", "#f58518"])
    axis.set_ylim(0, max(values) * 1.15)
    axis.set_ylabel("R²")
    axis.set_title("Train vs test: regression")
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_temporal_evaluation(test, test_pred, regression, output):
    """Plot chronological performance and normalized sensor context."""
    x = np.arange(len(test))
    actual = test[LABEL_COLUMN].to_numpy(dtype=float)
    predicted = np.asarray(test_pred, dtype=float)
    correct = actual == predicted if not regression else np.abs(actual - predicted) <= 0.5

    feature_columns = [column for column in test.columns
                       if column.endswith("_mean")][:6]
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                gridspec_kw={"height_ratios": [1, 2]})

    if regression:
        errors = np.abs(actual - predicted)
        rolling = pd.Series(errors).rolling(min(20, len(errors)), min_periods=1).mean()
        axes[0].plot(x, rolling, color="#d62728", linewidth=2)
        axes[0].set_ylabel("Rolling MAE")
        axes[0].set_title("Error over time (20-slot rolling MAE)")
    else:
        rolling = pd.Series(correct.astype(float)).rolling(
            min(20, len(correct)), min_periods=1).mean()
        axes[0].plot(x, rolling, color="#2ca02c", linewidth=2)
        axes[0].axhline(0.5, color="#888888", linestyle="--", linewidth=1)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy over time (20-slot rolling accuracy)")
    axes[0].grid(alpha=0.25)

    for column in feature_columns:
        values = test[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            low, high = np.nanmin(values), np.nanmax(values)
            if high > low:
                values = (values - low) / (high - low)
        axes[1].plot(x, values, linewidth=1, alpha=0.7, label=column.removesuffix("_mean"))

    for index in range(len(x) - 1):
        axes[1].axvspan(
            x[index], x[index + 1],
            color="#2ca02c" if correct[index] else "#d62728",
            alpha=0.10,
        )
    axes[1].step(x, actual, where="mid", color="#111111", linewidth=2,
                 label="recorded label")
    axes[1].step(x, predicted, where="mid", color="#ff7f0e", linewidth=1.5,
                 linestyle="--", label="model guess")
    axes[1].set_ylabel("Normalized sensors / level")
    axes[1].set_xlabel("Chronological held-out slots")
    axes[1].set_yticks(LEVELS)
    axes[1].set_ylim(-0.1, 3.1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left", ncol=3, fontsize="small")

    figure.suptitle("Held-out time-series evaluation")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def print_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Train an occupancy model")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--simulated", action="store_true", help="use included simulated data")
    source.add_argument("--api", action="store_true",
                        help="retrieve labelled data from the API (default)")
    parser.add_argument("--rooms", nargs="+", default=["1", "2"],
                        help="rooms to retrieve with --api (default: 1 2)")
    parser.add_argument("--start", help='inclusive start, "YYYY-MM-DD HH:MM"')
    parser.add_argument("--end", help='inclusive end, "YYYY-MM-DD HH:MM"')
    parser.add_argument("--days", type=int, default=30,
                        help="days to scan when dates are omitted (default: 30)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="fraction held out chronologically (default: 0.2)")
    parser.add_argument("--regression", action="store_true",
                        help="train Random Forest regression on the 0-3 occupancy value")
    args = parser.parse_args()

    if not args.simulated and (args.start is None) != (args.end is None):
        parser.error("--start and --end must be supplied together")
    if args.days < 1:
        parser.error("--days must be at least 1")
    if not 0 < args.test_size < 0.5:
        parser.error("--test-size must be between 0 and 0.5")

    if not args.simulated and not args.api:
        args.api = True
    if args.api and args.start is None:
        end = datetime.now().replace(second=0, microsecond=0)
        args.end = end.strftime("%Y-%m-%d %H:%M")
        args.start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M")
        print(f"Scanning the API by default: {args.start} -> {args.end}")

    config = load_config()
    sessions = load_sessions(args, config)
    if not sessions:
        raise SystemExit("No sessions found. Use --api or --simulated.")

    parts = []
    for name, data in sessions:
        windows = make_windows(data, config)
        if windows.empty:
            print(f"  {name}: no valid labelled windows")
            continue
        windows["session"] = name
        print(f"  {name}: {len(windows)} windows")
        parts.append(windows)
        if args.api:
            room = name.rsplit("-", 1)[-1]
            plot_raw_data(data, room, ROOT / "data" / "plots" / f"raw_room_{room}.png")
    if not parts:
        raise SystemExit("No valid labelled windows. Check that busy_level is present.")

    windows = pd.concat(parts).sort_index()
    train, test = split_windows(windows, args.test_size)
    X_train = train.drop(columns=[LABEL_COLUMN, "session"])
    X_test = test.drop(columns=[LABEL_COLUMN, "session"])
    y_train = train[LABEL_COLUMN]
    y_test = test[LABEL_COLUMN]
    fill_values = X_train.median()
    model = new_model(args.regression).fit(X_train.fillna(fill_values), y_train)
    train_pred = model.predict(X_train.fillna(fill_values))
    test_pred = model.predict(X_test.fillna(fill_values))

    if args.regression:
        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)
        model_path = ROOT / "models" / "regression_model.joblib"
    else:
        train_score = accuracy_score(y_train, train_pred)
        test_score = accuracy_score(y_test, test_pred)
        model_path = ROOT / config["model"]["path"]

    final_X = windows.drop(columns=[LABEL_COLUMN, "session"])
    final_fill_values = final_X.median()
    final_model = new_model(args.regression).fit(
        final_X.fillna(final_fill_values), windows[LABEL_COLUMN])
    plot_path = ROOT / "data" / "plots" / ("regression_evaluation.png"
                                           if args.regression else "classification_evaluation.png")
    temporal_plot_path = ROOT / "data" / "plots" / (
        "regression_timeseries.png" if args.regression else "classification_timeseries.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if args.regression:
        plot_regression_evaluation(train_score, test_score, plot_path)
    else:
        plot_classification_evaluation(train_score, test_score, y_test, test_pred, plot_path)
    plot_temporal_evaluation(test, test_pred, args.regression, temporal_plot_path)
    bundle = {"model": final_model, "features": list(final_X.columns),
              "fill_values": final_fill_values, "feature_min": final_X.min(),
              "feature_max": final_X.max(), "regression": args.regression}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\n  Evaluation plots saved:")
    print(f"    {plot_path}")
    print(f"    {temporal_plot_path}")
    print(f"  Model saved: {model_path}")


if __name__ == "__main__":
    main()

"""Classify occupancy in ten one-minute slots from one ten-minute retrieval."""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from occupancy.api_client import ApiSource, SimulatedSource, format_time
from occupancy.cleaning import clean
from occupancy.features import LABEL_COLUMN, build_windows
from occupancy.integration import merge_all
from occupancy.publish import publish_prediction
from occupancy.rnn import RecurrentRegressor
from occupancy.train import ROOT, load_config

AMSTERDAM = ZoneInfo("Europe/Amsterdam")
RETRIEVAL_MINUTES = 10
MAX_API_RANGE = timedelta(hours=12)
LEVEL_NAMES = {0: "quiet", 1: "slightly busy", 2: "busy", 3: "very busy"}
LEVEL_COLORS = {0: "32", 1: "33", 2: "35", 3: "31"}


def colour(text, code):
    """Add terminal colour when output is interactive."""
    return text if not sys.stdout.isatty() else f"\033[{code}m{text}\033[0m"


def display_time(timestamp):
    """Format an API timestamp for people rather than machines."""
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M")


def retrieve(source, start, end, config):
    """Retrieve and clean one inclusive time range."""
    data = {}
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + MAX_API_RANGE, end)
        chunk = source.fetch(chunk_start, chunk_end)
        if chunk:
            for sensor, readings in chunk.items():
                data.setdefault(sensor, []).extend(readings)
        chunk_start = chunk_end + timedelta(milliseconds=1)
    if not data:
        return None
    tables, pir, labels = clean(data, config["cleaning"])
    return merge_all(tables, pir, labels)


def classify(df, config, bundle, labelled_only=False):
    """Run all three models and return their predictions per valid slot."""
    windows = build_windows(
        df, config["windows"]["sensors"], config["windows"],
        config["cleaning"]["interval_seconds"],
        label_column=config["cleaning"]["label"]["sensor"],
    )
    if windows.empty:
        return []
    if labelled_only and LABEL_COLUMN in windows:
        windows = windows[windows[LABEL_COLUMN].notna()]
    if windows.empty:
        return []

    predictions = {}
    for name, model_bundle in bundle.items():
        X = windows.reindex(columns=model_bundle["features"]).fillna(
            model_bundle["fill_values"])
        predictions[name] = np.asarray(model_bundle["model"].predict(X)).round().clip(0, 3).astype(int)
    results = []
    for index, timestamp in enumerate(windows.index):
        levels = {name: int(values[index]) for name, values in predictions.items()}
        voted_level = max(
            levels.values(),
            key=lambda level: (sum(value == level for value in levels.values()),
                               levels["random_forest"] == level),
        )
        row = next(windows.iloc[[index]].itertuples())
        real_level = getattr(row, LABEL_COLUMN, None)
        results.append({
            "timestamp": format_time(timestamp),
            "predicted_level": voted_level,
            "random_forest_level": levels["random_forest"],
            "regression_level": levels["regression"],
            "rnn_level": levels["rnn"],
            "real_level": "" if real_level is None or real_level != real_level else int(real_level),
        })
    return results


def patch_predictions(source, results):
    """Patch consecutive slots with the same ensemble label."""
    if not results:
        return
    current_level = results[0]["predicted_level"]
    range_start = datetime.strptime(results[0]["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
    previous = range_start
    for result in results[1:] + [{"predicted_level": None, "timestamp": None}]:
        timestamp = (datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
                     if result["timestamp"] else None)
        if timestamp is not None and timestamp == previous + timedelta(minutes=1) and (
                result["predicted_level"] == current_level):
            previous = timestamp
            continue
        source.patch_prediction(
            range_start, previous + timedelta(minutes=1), current_level)
        if timestamp is not None:
            range_start = timestamp
            previous = timestamp
            current_level = result["predicted_level"]


def print_results(results, config, start, end, simulated):
    """Print a compact, readable classification report."""
    api = config["api"]
    print()
    print(colour("=" * 58, "36"))
    print(colour("  OCCUPANCY CLASSIFICATION", "1;36"))
    print(f"  Location : {api['building']} / room {api['room']}")
    print(f"  Mode     : {'SIMULATION' if simulated else 'LIVE API'}")
    print(f"  Period   : {display_time(format_time(start))} - {display_time(format_time(end))}")
    print(f"  Slots    : {len(results)} x 1 minute")
    print(colour("=" * 58, "36"))
    print("  Time   Prediction              Recorded level")
    print(colour("  " + "-" * 48, "90"))

    matches = 0
    for result in results:
        predicted = result["predicted_level"]
        prediction = f"{predicted} ({LEVEL_NAMES.get(predicted, 'unknown')})"
        recorded = result["real_level"]
        if recorded == "":
            recorded_text = "not available"
            line_colour = LEVEL_COLORS.get(predicted, "37")
        else:
            recorded_text = f"{recorded} ({LEVEL_NAMES.get(recorded, 'unknown')})"
            if predicted == recorded:
                matches += 1
                line_colour = "32"
            else:
                line_colour = "31"
        line = f"  {display_time(result['timestamp']):<6} {prediction:<30} {recorded_text}"
        print(colour(line, line_colour))

    print(colour("  " + "-" * 48, "90"))
    if any(result["real_level"] != "" for result in results):
        print(f"  Verification: {matches}/{len(results)} predictions match the recorded level")
    print(colour("=" * 58, "36"))
    print()


def main():
    parser = argparse.ArgumentParser(description="Classify occupancy in ten one-minute slots")
    parser.add_argument("--simulate", action="store_true",
                        help="use the simulated data instead of the API")
    parser.add_argument("--room", default=None,
                        help="Designlab room to query (default: config.yaml)")
    parser.add_argument("--start", help='historical start, "YYYY-MM-DD HH:MM"')
    parser.add_argument("--end", help='historical end, "YYYY-MM-DD HH:MM"')
    parser.add_argument("--without-patch", action="store_true",
                        help="save predictions without sending them to the API")
    args = parser.parse_args()

    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be supplied together")

    config = load_config()
    if args.room is not None:
        config["api"]["room"] = str(args.room)
    model_paths = {
        "random_forest": config["model"]["path"],
        "regression": config["model"]["regression_path"],
        "rnn": config["model"]["rnn_path"],
    }
    bundle = {name: joblib.load(ROOT / path) for name, path in model_paths.items()}

    if args.simulate:
        source = SimulatedSource(ROOT / "data" / "simulated" / "readings.json")
        first, last = source.time_span()
        if args.start:
            start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
            end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
            end = min(end, last)
        else:
            start = first
            end = min(start + timedelta(minutes=RETRIEVAL_MINUTES), last)
    else:
        source = ApiSource(config)
        if args.start:
            start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
            end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
        else:
            end = datetime.now(AMSTERDAM).replace(tzinfo=None)
            end -= timedelta(minutes=5, seconds=1)
            start = end - timedelta(minutes=RETRIEVAL_MINUTES)
        end = min(end, datetime.now(AMSTERDAM).replace(tzinfo=None))
    if end <= start:
        print(f"No existing readings between {format_time(start)} and {format_time(end)}.")
        return

    df = retrieve(source, start, end, config)
    if df is None:
        print(f"No readings between {format_time(start)} and {format_time(end)}.")
        return

    results = classify(df, config, bundle, labelled_only=bool(args.start))
    if not results:
        print(f"No complete one-minute slots between {format_time(start)} and {format_time(end)}.")
        return

    output = ROOT / config["output"]["predictions_path"]
    if args.simulate:
        output = output.with_name("predictions_simulated.csv")
    for result in results:
        publish_prediction(output, result)
    if not args.simulate and not args.without_patch:
        patch_predictions(source, results)
    print_results(results, config, start, end, args.simulate)


if __name__ == "__main__":
    main()

"""Classify occupancy in ten one-minute slots from one ten-minute retrieval."""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib

if __package__:
    from .api_client import ApiSource, SimulatedSource, format_time
    from .cleaning import clean
    from .features import LABEL_COLUMN, build_windows
    from .integration import merge_all
    from .publish import publish_prediction
    from .train import ROOT, load_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from occupancy.api_client import ApiSource, SimulatedSource, format_time
    from occupancy.cleaning import clean
    from occupancy.features import LABEL_COLUMN, build_windows
    from occupancy.integration import merge_all
    from occupancy.publish import publish_prediction
    from occupancy.train import ROOT, load_config

AMSTERDAM = ZoneInfo("Europe/Amsterdam")
RETRIEVAL_MINUTES = 10
LEVEL_NAMES = {
    0: "quiet",
    1: "slightly busy",
    2: "busy",
    3: "very busy",
}


def colour(text, code):
    """Add terminal colour when output is interactive."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def display_time(timestamp):
    """Format an API timestamp for people rather than machines."""
    moment = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    return moment.strftime("%H:%M")


def print_results(results, config, start, end, simulated):
    """Print a compact, readable classification report."""
    api = config["api"]
    location = f"{api['building']} / room {api['room']}"
    mode = "SIMULATION" if simulated else "LIVE API"

    print()
    print(colour("=" * 58, "36"))
    print(colour("  OCCUPANCY CLASSIFICATION", "1;36"))
    print(f"  Location : {location}")
    print(f"  Mode     : {mode}")
    print(f"  Period   : {display_time(format_time(start))} - "
          f"{display_time(format_time(end))}")
    print(f"  Slots    : {len(results)} x 1 minute")
    print(colour("=" * 58, "36"))
    print("  Time   Prediction              Recorded level")
    print(colour("  " + "-" * 48, "90"))

    matches = 0
    for result in results:
        predicted = result["predicted_level"]
        predicted_text = f"{predicted} ({LEVEL_NAMES.get(predicted, 'unknown')})"
        predicted_text = colour(predicted_text, {0: "32", 1: "33", 2: "35", 3: "31"}.get(predicted, "37"))
        recorded = result["real_level"]
        if recorded == "":
            recorded_text = "not available"
        else:
            recorded_text = f"{recorded} ({LEVEL_NAMES.get(recorded, 'unknown')})"
            if predicted == recorded:
                matches += 1
                recorded_text = colour(recorded_text, "32")
            else:
                recorded_text = colour(recorded_text, "31")
        print(f"  {display_time(result['timestamp']):<6} {predicted_text:<30} {recorded_text}")

    print(colour("  " + "-" * 48, "90"))
    if any(result["real_level"] != "" for result in results):
        print(f"  Verification: {matches}/{len(results)} predictions match the recorded level")
    print(colour("=" * 58, "36"))
    print()


def retrieve(source, start, end, config):
    """Retrieve and clean one inclusive time range."""
    data = source.fetch(start, end)
    if data is None:
        return None
    tables, pir, labels = clean(data, config["cleaning"])
    return merge_all(tables, pir, labels)


def classify(df, config, bundle):
    """Classify every valid one-minute slot and retain its label for checking."""
    windows = build_windows(
        df,
        config["windows"]["sensors"],
        config["windows"],
        config["cleaning"]["interval_seconds"],
        label_column=config["cleaning"]["label"]["sensor"],
    )
    if windows.empty:
        return []

    X = windows.reindex(columns=bundle["features"]).fillna(bundle["fill_values"])
    levels = bundle["model"].predict(X)
    results = []
    for timestamp, level, row in zip(windows.index, levels, windows.itertuples()):
        real_level = getattr(row, LABEL_COLUMN, None)
        results.append({
            "timestamp": format_time(timestamp),
            "predicted_level": int(level),
            "real_level": "" if real_level is None or real_level != real_level else int(real_level),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify occupancy in ten one-minute slots")
    parser.add_argument("--simulate", action="store_true",
                        help="use the simulated data instead of the API")
    args = parser.parse_args()

    config = load_config()
    bundle = joblib.load(ROOT / config["model"]["path"])

    if args.simulate:
        source = SimulatedSource(ROOT / "data" / "simulated" / "readings.json")
        first, last = source.time_span()
        start = first
        end = min(start + timedelta(minutes=RETRIEVAL_MINUTES), last)
    else:
        source = ApiSource(config)
        end = datetime.now(AMSTERDAM).replace(tzinfo=None)
        start = end - timedelta(minutes=RETRIEVAL_MINUTES)

    df = retrieve(source, start, end, config)
    if df is None:
        print(f"No readings between {format_time(start)} and {format_time(end)}.")
        return

    results = classify(df, config, bundle)
    if not results:
        print(f"No complete one-minute slots between {format_time(start)} and {format_time(end)}.")
        return

    output = ROOT / config["output"]["predictions_path"]
    if args.simulate:
        output = output.with_name("predictions_simulated.csv")
    for result in results:
        publish_prediction(output, result)
    print_results(results, config, start, end, args.simulate)


if __name__ == "__main__":
    main()

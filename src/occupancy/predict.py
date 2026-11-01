"""Live prediction.

Every 'step_seconds' (30 s): requests the last few minutes of data from the
API, prepares them exactly as in training, predicts the occupancy level (0-3)
and saves it to the predictions CSV.

The window ends at the time of the last reading returned by the API, not at
the laptop's clock: so it does not matter if the clocks of the laptop and the
Raspberry Pi do not match.

Run from the repository root (Windows PowerShell):
    $env:PYTHONPATH = "src"
    python -m occupancy.predict              with the real API
    python -m occupancy.predict --simulate   with the simulated data (no API)
"""
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib

from occupancy.api_client import ApiSource, SimulatedSource, format_time
from occupancy.cleaning import clean
from occupancy.features import LABEL_COLUMN, build_windows
from occupancy.integration import merge_all
from occupancy.publish import publish_prediction
from occupancy.train import ROOT, load_config

AMSTERDAM = ZoneInfo("Europe/Amsterdam")


def predict_at(now, source, config, bundle, clock_margin, last_seen=None):
    """Prediction for the last window with data. Returns a row or None.

    Data is requested around 'now' with a margin in case the clocks do not
    match, and the window ends at the last reading received.
    If that reading is the same as in the previous prediction ('last_seen'),
    there is no new data and the prediction is not repeated.
    """
    pred_cfg = config["prediction"]

    # ================================================================
    # 1. GET THE DATA (from the real API or the simulated data)
    #    source.fetch is in api_client.py: it makes the GET request and
    #    returns the "data" field of the JSON, or None if there is no data (404).
    # ================================================================
    # More minutes than the window are requested: the cleaning needs earlier
    # data (smoothing, gaps) to handle the start of the window properly.
    history = timedelta(minutes=config["windows"]["size_minutes"] + pred_cfg["extra_history_minutes"])
    data = source.fetch(now - history - clock_margin, now + clock_margin)
    if data is None:
        print(format_time(now), "no data from the API")
        return None

    # ================================================================
    # 2. RUN THE DATA THROUGH THE PHASES (the same functions as in train.py)
    # ================================================================
    # Phase 1, cleaning (cleaning.py): duplicates, range, spikes, gaps and smoothing
    tables, pir, labels = clean(data, config["cleaning"])
    # Phase 2, integration (integration.py): everything in one table, one row every 10 s
    df = merge_all(tables, pir, labels)
    # Phase 4, windows and features (features.py): only the last window
    windows = build_windows(df, config["windows"]["sensors"], config["windows"],
                            config["cleaning"]["interval_seconds"],
                            label_column=config["cleaning"]["label"]["sensor"], last_only=True)

    # Checks: window with too little data, or the same as the previous prediction
    if windows.empty:
        print(format_time(now), "not enough data in the last window")
        return None
    if last_seen is not None and windows.index[0] <= last_seen:
        print(format_time(now), "no new data since", format_time(last_seen))
        return None

    # ================================================================
    # 3. USE THE MODEL
    # ================================================================
    # True label (if the API sends it): only for comparison, it does NOT go into the model
    real_level = windows[LABEL_COLUMN].iloc[0] if LABEL_COLUMN in windows else None
    # Model input: the same features, in the same order and with the same
    # filling of empty values as in training
    X = windows.reindex(columns=bundle["features"])
    # Empty features (a sensor without data): they are filled in, with a warning
    missing = list(X.columns[X.iloc[0].isna()])
    X = X.fillna(bundle["fill_values"])

    # The model is called here: each tree votes and we get the probability of each level
    probabilities = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["model"].classes_)
    level = classes[probabilities.argmax()]
    confidence = probabilities.max()

    # Unreliable prediction: the model is unsure, a sensor is missing, or some
    # value was never seen in training
    reliable = confidence >= pred_cfg["min_confidence"] and not missing
    if missing:
        print(format_time(now), "warning: no data for", ", ".join(missing))
    if "feature_min" in bundle:
        outside = (X.iloc[0] < bundle["feature_min"]) | (X.iloc[0] > bundle["feature_max"])
        reliable = reliable and not outside.any()

    # ================================================================
    # 4. BUILD THE RESULT (main saves it to the CSV with publish_prediction)
    # ================================================================
    row = {"window_end": windows.index[0],
           "timestamp": format_time(windows.index[0]),
           "predicted_level": int(level),
           "confidence": round(float(confidence), 3),
           "reliable": bool(reliable),
           "real_level": "" if real_level is None or real_level != real_level else int(real_level)}
    for cls, p in zip(classes, probabilities):
        row[f"prob_{int(cls)}"] = round(float(p), 3)
    return row


def main():
    parser = argparse.ArgumentParser(description="Live prediction of the occupancy level")
    parser.add_argument("--simulate", action="store_true",
                        help="use the simulated data instead of the API (no waiting between steps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="number of predictions before stopping (default: run forever)")
    args = parser.parse_args()

    # ---------- Setup, once at start-up ----------
    config = load_config()
    # Load the trained model (saved by train.py in models/model.joblib)
    bundle = joblib.load(ROOT / config["model"]["path"])
    output = ROOT / config["output"]["predictions_path"]
    step = timedelta(seconds=config["windows"]["step_seconds"])

    # Choose where the data comes from: simulated (--simulate) or real API
    if args.simulate:
        source = SimulatedSource(ROOT / "data" / "simulated" / "readings.json")
        first, last = source.time_span()
        now = first + timedelta(minutes=config["windows"]["size_minutes"]
                                + config["prediction"]["extra_history_minutes"])
        output = output.with_name("predictions_simulated.csv")
        clock_margin = timedelta(0)        # in the simulation there are no different clocks
    else:
        source = ApiSource(config)
        clock_margin = timedelta(minutes=config["prediction"]["clock_margin_minutes"])

    last_seen = None                       # end of the last window already predicted

    # ---------- Loop: one prediction every 30 s ----------
    done = 0
    while args.steps is None or done < args.steps:
        if not args.simulate:
            # Current Amsterdam time, without time zone (as the API expects it)
            now = datetime.now(AMSTERDAM).replace(tzinfo=None)
        elif now > last:
            break

        # Get data, run it through the phases and predict (all in predict_at)
        try:
            row = predict_at(now, source, config, bundle, clock_margin, last_seen)
        except Exception as error:          # the API can fail: warn and carry on
            print(format_time(now), "error:", error)
            row = None

        # Save the prediction to the CSV (publish.py) and show it on screen
        if row is not None:
            last_seen = row["window_end"]
            publish_prediction(output, row)
            print(row["timestamp"], "level", row["predicted_level"],
                  "confidence", row["confidence"],
                  "" if row["reliable"] else "(unreliable)",
                  f"true {row['real_level']}" if row["real_level"] != "" else "")

        # Wait until the next prediction
        done += 1
        if args.simulate:
            now += step
        else:
            time.sleep(step.total_seconds())


if __name__ == "__main__":
    main()

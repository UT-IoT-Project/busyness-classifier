"""Model training (this is the code from notebooks/04_modelo.ipynb).

Run from the repository root (Windows PowerShell):
    $env:PYTHONPATH = "src"
    python -m occupancy.train               with the real sessions in data/raw/
    python -m occupancy.train --simulated   with the simulated data

With real data, each file in data/raw/ is one session (see download.py).
Validation trains on all sessions but one and tests on the one left out,
once per session: this measures whether the model works in conditions it
has not seen.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from occupancy.cleaning import clean
from occupancy.features import LABEL_COLUMN, build_windows
from occupancy.integration import merge_all

ROOT = Path(__file__).resolve().parents[2]
LEVELS = [0, 1, 2, 3]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sessions(simulated=False):
    """Return a list of (session name, 'data' field of the API response)."""
    if simulated:
        with open(ROOT / "data" / "simulated" / "readings.json", encoding="utf-8") as f:
            return [("simulated", json.load(f)["data"])]
    sessions = []
    for path in sorted((ROOT / "data" / "raw").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            sessions.append((path.stem, json.load(f)["data"]))
    return sessions


def make_windows(data, config):
    """Phases 1, 2 and 4: cleaning, integration and windows.

    These are the same functions used by predict.py, so the model sees live
    data in the same shape it learned from.
    """
    # Phase 1, cleaning (cleaning.py)
    tables, pir, labels = clean(data, config["cleaning"])
    # Phase 2, integration (integration.py)
    df = merge_all(tables, pir, labels)
    # Phase 4, windows and features (features.py): all windows
    return build_windows(df, config["windows"]["sensors"], config["windows"],
                         config["cleaning"]["interval_seconds"],
                         label_column=config["cleaning"]["label"]["sensor"])


def new_model():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)


def validate(X, y, groups, splitter):
    """Train and test on each split; return true levels, predicted levels and group."""
    all_true, all_pred, all_group = [], [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        fill_values = X_train.median()
        model = new_model().fit(X_train.fillna(fill_values), y.iloc[train_idx])
        all_true.extend(y.iloc[test_idx])
        all_pred.extend(model.predict(X_test.fillna(fill_values)))
        all_group.extend(groups[test_idx])
    return np.array(all_true), np.array(all_pred), np.array(all_group)


def report(y_true, y_pred, groups, by_group=False):
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print("Within one level:", round(np.mean(np.abs(y_true - y_pred) <= 1), 3))
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LEVELS),
                       index=[f"true {i}" for i in LEVELS],
                       columns=[f"pred {i}" for i in LEVELS]))
    print(classification_report(y_true, y_pred, labels=LEVELS, digits=2, zero_division=0))
    if by_group:
        print("Accuracy on each session left out:")
        for group in sorted(set(groups)):
            mask = groups == group
            print(f"  {group}: {accuracy_score(y_true[mask], y_pred[mask]):.3f} ({mask.sum()} windows)")


def train_final(X, y):
    """Final model trained on all data, together with what is needed to use it live."""
    fill_values = X.median()
    model = new_model().fit(X.fillna(fill_values), y)
    return {"model": model,
            "features": list(X.columns),
            "fill_values": fill_values,
            "feature_min": X.min(),       # range seen in training: used to warn
            "feature_max": X.max()}       # when something never seen appears live


def show_importances(model, features):
    """Phase 3, reduction: how much each feature is used by the model (they add up to 1).

    Features close to zero are candidates to remove; check afterwards that
    validation accuracy does not drop without them.
    """
    importances = pd.Series(model.feature_importances_, index=features)
    print("\nFeature importance (phase 3, reduction):")
    for name, value in importances.sort_values(ascending=False).items():
        print(f"  {name:<22} {value:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the occupancy model")
    parser.add_argument("--simulated", action="store_true", help="use the simulated data")
    args = parser.parse_args()

    config = load_config()
    sessions = load_sessions(args.simulated)
    if not sessions:
        raise SystemExit("No sessions in data/raw/. Download them with occupancy.download "
                         "or use --simulated to test with the simulated data.")

    # Windows of each session separately (times from different sessions are not mixed)
    parts = []
    for name, data in sessions:
        windows = make_windows(data, config)
        windows["session"] = name
        print(f"Session {name}: {len(windows)} windows,",
              windows[LABEL_COLUMN].value_counts().sort_index().to_dict() if len(windows) else "none")
        parts.append(windows)
    windows = pd.concat(parts)
    if windows.empty:
        raise SystemExit("No valid labelled window: check that the sessions have busy_level")

    X = windows.drop(columns=[LABEL_COLUMN, "session"])
    y = windows[LABEL_COLUMN]

    missing_levels = sorted(set(LEVELS) - set(y.unique()))
    if missing_levels:
        print("WARNING: there are no examples of levels", missing_levels,
              "- the model will not be able to predict them")

    # Validation: leave one whole session out. With a single session (for
    # example, the simulated data) 15-minute blocks are used instead.
    if len(sessions) >= 2:
        print("\nValidation leaving one session out each time")
        for name in windows["session"].unique():
            absent = sorted(set(LEVELS) - set(windows.loc[windows["session"] == name, LABEL_COLUMN]))
            if absent:
                print(f"WARNING: session {name} has no levels {absent}; "
                      "each session should go through all 4 levels")
        groups = windows["session"].to_numpy()
        splitter = LeaveOneGroupOut()
    else:
        print("\nWARNING: only one session; validating with 15-minute blocks "
              "(less reliable than leaving a session out)")
        groups = np.array(windows.index.floor("15min").astype(str))
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    report(*validate(X, y, groups, splitter), by_group=len(sessions) >= 2)

    bundle = train_final(X, y)
    show_importances(bundle["model"], bundle["features"])
    path = ROOT / config["model"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print("Model saved to", path)


if __name__ == "__main__":
    main()

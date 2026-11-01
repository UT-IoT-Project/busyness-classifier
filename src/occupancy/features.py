"""Phase 4: transformation. Turns the integrated table into one row per window.

This is the code from notebooks/03_caracteristicas.ipynb.
Windows only look backwards, so they work the same way live.

Tolerance to sensor failures:
- If a sensor has no data, its features are left empty (NaN) and the window is
  still valid as long as most sensors do have data. When predicting, the empty
  values are filled in and the prediction is marked as unreliable.
- PIRs are found by name (every 'pir-<ID>' column), however many there are.
"""
import numpy as np
import pandas as pd

LABEL_COLUMN = "label"


def most_frequent(values):
    """Most frequent value in a window, ignoring empty values."""
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def pir_columns(df):
    """PIR columns present in the table (pir-1, pir-2...)."""
    return [column for column in df.columns if str(column).startswith("pir-")]


def build_windows(df, sensors, window_cfg, interval_seconds,
                  label_column="busy_level", last_only=False):
    """Return a table with one row per valid window.

    - One window of 'size_minutes' every 'step_seconds'.
    - With last_only=True only the last window is returned (live prediction);
      if it is not valid, an empty table is returned.
    - Columns: the features and, if df has a label, 'label'.
    - A window is valid if it is complete, if at least 'min_sensor_fraction'
      of the sensors have 'min_coverage' of data and, when training, if it has a label.
    """
    window = f"{window_cfg['size_minutes']}min"
    step = window_cfg["step_seconds"] // interval_seconds
    min_coverage = window_cfg.get("min_coverage", 0.8)
    min_sensor_fraction = window_cfg.get("min_sensor_fraction", 0.5)

    # Missing columns (sensor without data) are created empty
    df = df.reindex(columns=list(dict.fromkeys(list(df.columns) + list(sensors))))
    roll = df.rolling(window, min_periods=1)

    # Features
    features = pd.DataFrame(index=df.index)
    for sensor in sensors:
        features[f"{sensor}_mean"] = roll[sensor].mean()

    features["co2Equivalent_change"] = (
        df["co2Equivalent"] - df["co2Equivalent"].shift(freq=window)).reindex(df.index)
    features["mic_vol_max"] = roll["mic_vol"].max()
    features["mic_vol_std"] = roll["mic_vol"].std()

    pirs = pir_columns(df)
    if pirs:
        features["pir_mean"] = df[pirs].mean(axis=1).rolling(window).mean()
        features["pir_max"] = df[pirs].max(axis=1).rolling(window).max()
    else:
        features["pir_mean"] = np.nan
        features["pir_max"] = np.nan

    # Label of each window: the most frequent level
    has_label = label_column in df.columns and df[label_column].notna().any()
    if has_label:
        features[LABEL_COLUMN] = df[label_column].rolling(window).apply(
            most_frequent, raw=True)

    # Coverage: share of the window with data, sensor by sensor
    sensor_coverage = df[sensors].notna().rolling(window).mean()
    enough_sensors = (sensor_coverage >= min_coverage).mean(axis=1) >= min_sensor_fraction

    # One window every 'step_seconds' (or only the last one, live)
    if last_only:
        windows = features.iloc[[-1]]
        enough_sensors = enough_sensors.iloc[[-1]]
    else:
        windows = features.iloc[::step]
        enough_sensors = enough_sensors.iloc[::step]

    valid = (windows.index >= df.index[0] + pd.Timedelta(window)) & enough_sensors
    if has_label and not last_only:
        valid = valid & windows[LABEL_COLUMN].notna()

    windows = windows[valid].copy()
    if has_label and not last_only:
        windows[LABEL_COLUMN] = windows[LABEL_COLUMN].astype(int)
    return windows

"""Phase 1: data cleaning.

This is the code from notebooks/01_limpieza.ipynb, turned into functions so that
training and live prediction clean the data in exactly the same way.
All settings are read from the "cleaning" section of config.yaml.

Order of the steps (see the pipeline document):
    0. Turn the API response into tables
    1.1 Duplicates
    1.2 Range check
    1.3 Isolated spikes
    1.4 Gaps (common time grid + interpolate short gaps)
    1.5 Smoothing (optional)

The label (busy_level, 0 to 3, set by a person) is kept apart from the sensors:
it only goes through duplicates, range check and the time grid. It is never
interpolated or smoothed, because an in-between level does not exist.

Usage:
    from occupancy.cleaning import clean
    tables, pir, labels = clean(data, config["cleaning"])
"""
import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
PIR_SENSOR = "pirs"            # in the API: {timestamp, values: [{ID, activity}]}
DEFAULT_LABEL = "busy_level"


# ---------- 0. From the API response to tables ----------

def readings_to_tables(data, label=DEFAULT_LABEL):
    """Turn the 'data' field of the API response into pandas tables.

    Returns:
    - tables: dict {sensor: table with columns timestamp and value}
    - pir: one table with timestamp and one column per PIR (pir-1, pir-2...),
      holding the activity of each one (0 to 100)
    The PIRs and the label are not included in tables.
    """
    tables = {}
    for sensor, readings in data.items():
        if sensor in (PIR_SENSOR, label):
            continue
        table = pd.DataFrame(readings, columns=["timestamp", "value"])
        table["timestamp"] = pd.to_datetime(table["timestamp"], format=TIME_FORMAT)
        tables[sensor] = table

    rows = []
    for reading in data.get(PIR_SENSOR, []):
        row = {"timestamp": reading["timestamp"]}
        for p in reading["values"]:
            row[f"pir-{p['ID']}"] = p["activity"]
        rows.append(row)
    pir = pd.DataFrame(rows)
    if not pir.empty:
        pir["timestamp"] = pd.to_datetime(pir["timestamp"], format=TIME_FORMAT)

    return tables, pir


# ---------- 1.1 Duplicates ----------

def remove_duplicates(table):
    """Remove rows that are exactly equal (MQTT QoS 1 can deliver a message twice).

    Compares the whole row, not only the timestamp.
    """
    return table.drop_duplicates()


# ---------- 1.2 Range check ----------

def check_range(table, limits):
    """Empty (NaN) the values outside [min, max], without deleting the row."""
    table = table.copy()
    out_of_range = (table["value"] < limits["min"]) | (table["value"] > limits["max"])
    table["value"] = table["value"].mask(out_of_range)
    return table


# ---------- 1.3 Isolated spikes ----------

def remove_spikes(table, threshold, window_readings):
    """Empty the values that are further than 'threshold' from the median of their neighbours."""
    table = table.copy()
    median = table["value"].rolling(window_readings, center=True, min_periods=1).median()
    distance = table["value"].sub(median).abs()
    table["value"] = table["value"].mask(distance > threshold)
    return table


# ---------- 1.4 Gaps ----------

def to_grid(table, interval):
    """Put a table on the regular time grid.

    Returns a table with timestamp as index and exactly one row every 'interval'.
    Moments without a reading become NaN. Also works for the PIR table,
    which has several columns.
    """
    return table.set_index("timestamp").resample(interval).mean()


def fill_short_gaps(series, max_gap_readings):
    """Interpolate only the gaps of at most max_gap_readings consecutive readings.

    Longer gaps are left completely empty (not just partly filled at the start).
    """
    empty = series.isna()
    group = (empty != empty.shift()).cumsum()
    gap_length = empty.groupby(group).transform("sum")
    short_gap = empty & (gap_length <= max_gap_readings)
    filled = series.interpolate(method="time", limit_area="inside")
    return series.mask(short_gap, filled)


# ---------- 1.5 Smoothing ----------

def smooth(series, window):
    """Backward-looking moving average; does not fill gaps that were already empty."""
    smoothed = series.rolling(window, min_periods=1).mean()
    return smoothed.where(series.notna())


# ---------- Label ----------

def clean_labels(data, label_cfg, interval):
    """Clean the label (busy_level) and put it on the time grid.

    Removes duplicates, empties values outside [min, max] and keeps the last
    value of each grid cell. Gaps are left empty: those rows cannot be used
    for training.
    Returns a table with timestamp and the label column.
    """
    name = label_cfg["sensor"]
    table = pd.DataFrame(data.get(name, []), columns=["timestamp", "value"])
    table["timestamp"] = pd.to_datetime(table["timestamp"], format=TIME_FORMAT)
    table = check_range(remove_duplicates(table), label_cfg)
    series = table.set_index("timestamp")["value"].resample(interval).last()
    return series.reset_index(name=name)


# ---------- Whole process ----------

def clean(data, cfg):
    """Apply the whole cleaning to the 'data' field of the API response.

    cfg is the "cleaning" section of config.yaml.
    Returns (tables, pir, labels), all on the same time grid.
    labels is None if the response has no label (for example, live).
    """
    tables, pir = readings_to_tables(data, cfg["label"]["sensor"])
    interval = pd.Timedelta(seconds=cfg["interval_seconds"])
    max_gap = cfg["max_gap_readings"]
    smoothing = cfg["smoothing"]
    smooth_window = pd.Timedelta(seconds=smoothing["window_seconds"])

    for sensor in list(tables):
        table = remove_duplicates(tables[sensor])
        if sensor in cfg["limits"]:
            table = check_range(table, cfg["limits"][sensor])
        if sensor in cfg["spike_thresholds"]:
            table = remove_spikes(table, cfg["spike_thresholds"][sensor],
                                  cfg["spike_window_readings"])

        series = to_grid(table, interval)["value"]
        series = fill_short_gaps(series, max_gap)
        if smoothing["enabled"] and sensor in smoothing["sensors"]:
            series = smooth(series, smooth_window)

        tables[sensor] = series.reset_index(name="value")

    if not pir.empty:
        grid = to_grid(remove_duplicates(pir), interval)
        for column in grid.columns:
            grid[column] = fill_short_gaps(grid[column], max_gap)
        pir = grid.reset_index()

    labels = None
    if data.get(cfg["label"]["sensor"]):
        labels = clean_labels(data, cfg["label"], interval)

    return tables, pir, labels

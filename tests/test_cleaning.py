"""Tests of the cleaning with the simulated data (data/simulated/).

Run from the repository root:  python -m pytest
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from occupancy.cleaning import clean

ROOT = Path(__file__).resolve().parents[1]


def run_cleaning():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["cleaning"]
    with open(ROOT / "data" / "simulated" / "readings.json", encoding="utf-8") as f:
        data = json.load(f)["data"]
    return clean(data, cfg)


def test_all_sensors_share_the_same_time_grid():
    tables, pir, labels = run_cleaning()
    reference = tables["temperature"]["timestamp"]
    assert len(reference) == 1080
    for table in tables.values():
        assert table["timestamp"].equals(reference)
    assert pir["timestamp"].equals(reference)


def test_only_the_long_gap_stays_empty():
    tables, pir, labels = run_cleaning()
    for table in tables.values():
        assert table["value"].isna().sum() == 18
    for column in ["pir-1", "pir-2", "pir-3"]:
        assert pir[column].isna().sum() == 18


def test_no_impossible_values_or_spikes_remain():
    tables, _, _ = run_cleaning()
    assert tables["co2Equivalent"]["value"].min() >= 0
    assert tables["temperature"]["value"].max() <= 50
    assert tables["humidity"]["value"].max() <= 100
    assert tables["mic_vol"]["value"].max() < -15   # the -5 dB spikes are gone


def test_labels_are_levels_on_the_same_grid():
    tables, _, labels = run_cleaning()
    assert labels["timestamp"].equals(tables["temperature"]["timestamp"])
    assert set(labels["busy_level"].dropna().unique()) <= {0, 1, 2, 3}
    # unfilled gaps: the long one (18) and the 3 single lost messages
    assert labels["busy_level"].isna().sum() == 21
    assert "busy_level" not in tables

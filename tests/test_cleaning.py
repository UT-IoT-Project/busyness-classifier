"""Pruebas de la limpieza con los datos simulados (data/simulated/).

Ejecutar desde la raiz del repositorio:  python -m pytest
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
    assert tables["mic_vol"]["value"].max() < 0.5


def test_labels_are_levels_on_the_same_grid():
    tables, _, labels = run_cleaning()
    assert labels["timestamp"].equals(tables["temperature"]["timestamp"])
    assert set(labels["busynessLabel"].dropna().unique()) <= {0, 1, 2, 3}
    # huecos sin rellenar: el largo (18) y los 3 mensajes sueltos perdidos
    assert labels["busynessLabel"].isna().sum() == 21
    assert "busynessLabel" not in tables

"""Pruebas de las ventanas con los datos simulados, incluidos fallos de sensores.

Ejecutar desde la raiz del repositorio:  python -m pytest
"""
import json
from pathlib import Path

import yaml

from occupancy.cleaning import clean
from occupancy.features import LABEL_COLUMN, build_windows
from occupancy.integration import merge_all

ROOT = Path(__file__).resolve().parents[1]


def load():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(ROOT / "data" / "simulated" / "readings.json", encoding="utf-8") as f:
        data = json.load(f)["data"]
    return config, data


def windows_for(data, config, last_only=False):
    tables, pir, labels = clean(data, config["cleaning"])
    df = merge_all(tables, pir, labels)
    return build_windows(df, config["windows"]["sensors"], config["windows"],
                         config["cleaning"]["interval_seconds"], last_only=last_only)


def test_training_windows():
    config, data = load()
    windows = windows_for(data, config)
    assert len(windows) == 345
    assert set(windows[LABEL_COLUMN].unique()) <= {0, 1, 2, 3}
    assert "busy_level" not in windows.columns


def test_live_window_without_label():
    config, data = load()
    data.pop("busy_level")
    windows = windows_for(data, config, last_only=True)
    assert len(windows) == 1
    assert LABEL_COLUMN not in windows.columns


def test_one_sensor_without_data_still_gives_windows():
    config, data = load()
    data["co2Equivalent"] = []
    windows = windows_for(data, config)
    assert len(windows) > 300
    assert windows["co2Equivalent_mean"].isna().all()


def test_two_pirs_instead_of_three():
    config, data = load()
    for reading in data["pirs"]:
        reading["values"] = reading["values"][:2]
    windows = windows_for(data, config)
    assert len(windows) == 345
    assert windows["pir_mean"].notna().all()

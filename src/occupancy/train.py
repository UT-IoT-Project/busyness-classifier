"""Entrenamiento del modelo (es el codigo del notebook 04_modelo.ipynb).

Ejecutar desde la raiz del repositorio (en Windows):
    set PYTHONPATH=src
    python -m occupancy.train               con las sesiones reales de data/raw/
    python -m occupancy.train --simulated   con los datos simulados

Con datos reales, cada archivo de data/raw/ es una sesion (ver download.py).
La validacion entrena con todas las sesiones menos una y prueba con la que
queda fuera, una vez por sesion: asi se mide si el modelo funciona en
condiciones que no ha visto.
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
    """Devuelve una lista de (nombre de la sesion, campo 'data' de la API)."""
    if simulated:
        with open(ROOT / "data" / "simulated" / "readings.json", encoding="utf-8") as f:
            return [("simulada", json.load(f)["data"])]
    sessions = []
    for path in sorted((ROOT / "data" / "raw").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            sessions.append((path.stem, json.load(f)["data"]))
    return sessions


def make_windows(data, config):
    """Fases 1, 2 y 4: limpieza, integracion y ventanas.

    Son las mismas funciones que usa predict.py, para que el modelo vea en vivo
    los datos con la misma forma con la que aprendio.
    """
    # Fase 1, limpieza (cleaning.py)
    tables, pir, labels = clean(data, config["cleaning"])
    # Fase 2, integracion (integration.py)
    df = merge_all(tables, pir, labels)
    # Fase 4, ventanas y caracteristicas (features.py): todas las ventanas
    return build_windows(df, config["windows"]["sensors"], config["windows"],
                         config["cleaning"]["interval_seconds"],
                         label_column=config["cleaning"]["label"]["sensor"])


def new_model():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)


def validate(X, y, groups, splitter):
    """Entrena y prueba en cada particion; devuelve niveles reales, predichos y grupo."""
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
    print("Exactitud:", round(accuracy_score(y_true, y_pred), 3))
    print("Acierto con margen de un nivel:", round(np.mean(np.abs(y_true - y_pred) <= 1), 3))
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LEVELS),
                       index=[f"real {i}" for i in LEVELS],
                       columns=[f"pred {i}" for i in LEVELS]))
    print(classification_report(y_true, y_pred, labels=LEVELS, digits=2, zero_division=0))
    if by_group:
        print("Exactitud en cada sesion dejada fuera:")
        for group in sorted(set(groups)):
            mask = groups == group
            print(f"  {group}: {accuracy_score(y_true[mask], y_pred[mask]):.3f} ({mask.sum()} ventanas)")


def train_final(X, y):
    """Modelo final con todos los datos, junto con lo necesario para usarlo en vivo."""
    fill_values = X.median()
    model = new_model().fit(X.fillna(fill_values), y)
    return {"model": model,
            "features": list(X.columns),
            "fill_values": fill_values,
            "feature_min": X.min(),       # rango visto al entrenar: sirve para avisar
            "feature_max": X.max()}       # si en vivo aparece algo nunca visto


def main():
    parser = argparse.ArgumentParser(description="Entrenar el modelo de ocupacion")
    parser.add_argument("--simulated", action="store_true", help="usar los datos simulados")
    args = parser.parse_args()

    config = load_config()
    sessions = load_sessions(args.simulated)
    if not sessions:
        raise SystemExit("No hay sesiones en data/raw/. Descargalas con occupancy.download "
                         "o usa --simulated para probar con los datos simulados.")

    # Ventanas de cada sesion por separado (no se mezclan horas de sesiones distintas)
    parts = []
    for name, data in sessions:
        windows = make_windows(data, config)
        windows["session"] = name
        print(f"Sesion {name}: {len(windows)} ventanas,",
              windows[LABEL_COLUMN].value_counts().sort_index().to_dict() if len(windows) else "ninguna")
        parts.append(windows)
    windows = pd.concat(parts)
    if windows.empty:
        raise SystemExit("Ninguna ventana valida con etiqueta: revisa que las sesiones tengan busy_level")

    X = windows.drop(columns=[LABEL_COLUMN, "session"])
    y = windows[LABEL_COLUMN]

    missing_levels = sorted(set(LEVELS) - set(y.unique()))
    if missing_levels:
        print("AVISO: no hay ningun ejemplo de los niveles", missing_levels,
              "- el modelo no podra predecirlos")

    # Validacion: dejar fuera una sesion entera. Con una sola sesion (por
    # ejemplo, los datos simulados) se usan bloques de 15 minutos.
    if len(sessions) >= 2:
        print("\nValidacion dejando fuera una sesion cada vez")
        for name in windows["session"].unique():
            absent = sorted(set(LEVELS) - set(windows.loc[windows["session"] == name, LABEL_COLUMN]))
            if absent:
                print(f"AVISO: la sesion {name} no tiene los niveles {absent}; "
                      "conviene que cada sesion recorra los 4 niveles")
        groups = windows["session"].to_numpy()
        splitter = LeaveOneGroupOut()
    else:
        print("\nAVISO: solo hay una sesion; se valida con bloques de 15 minutos "
              "(menos fiable que dejar fuera una sesion)")
        groups = np.array(windows.index.floor("15min").astype(str))
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    report(*validate(X, y, groups, splitter), by_group=len(sessions) >= 2)

    bundle = train_final(X, y)
    path = ROOT / config["model"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print("Modelo guardado en", path)


if __name__ == "__main__":
    main()

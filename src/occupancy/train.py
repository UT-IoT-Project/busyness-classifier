"""Entrenamiento del modelo (es el codigo del notebook 04_modelo.ipynb).

Ejecutar desde la raiz del repositorio (en Windows):
    set PYTHONPATH=src
    python -m occupancy.train

Por ahora entrena con los datos simulados. Con datos reales, cambia
load_data() para leer los archivos guardados en data/raw/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from occupancy.cleaning import clean
from occupancy.features import LABEL_COLUMN, build_windows
from occupancy.integration import merge_all

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    """Datos de entrenamiento (campo 'data' de la API). Ahora: los simulados."""
    with open(ROOT / "data" / "simulated" / "readings.json", encoding="utf-8") as f:
        return json.load(f)["data"]


def make_windows(data, config):
    """Fases 1, 2 y 4: limpieza, integracion y ventanas."""
    tables, pir, labels = clean(data, config["cleaning"])
    df = merge_all(tables, pir, labels)
    return build_windows(df, list(tables), config["windows"],
                         config["cleaning"]["interval_seconds"])


def new_model():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)


def validate(X, y, groups, n_splits=4):
    """Validacion por grupos: devuelve los niveles reales y predichos de todas las particiones."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        fill_values = X_train.median()
        model = new_model().fit(X_train.fillna(fill_values), y.iloc[train_idx])
        all_true.extend(y.iloc[test_idx])
        all_pred.extend(model.predict(X_test.fillna(fill_values)))
    return np.array(all_true), np.array(all_pred)


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
    config = load_config()
    windows = make_windows(load_data(), config)
    X = windows.drop(columns=LABEL_COLUMN)
    y = windows[LABEL_COLUMN]

    # Con datos simulados (una sola sesion): bloques de 15 min.
    # Con datos reales: el grupo debe ser la sesion de recogida.
    groups = windows.index.floor("15min")
    y_true, y_pred = validate(X, y, groups)
    print("Ventanas:", len(windows))
    print("Exactitud:", round(accuracy_score(y_true, y_pred), 3))
    print("Acierto con margen de un nivel:", round(np.mean(np.abs(y_true - y_pred) <= 1), 3))
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]),
                       index=[f"real {i}" for i in range(4)],
                       columns=[f"pred {i}" for i in range(4)]))

    bundle = train_final(X, y)
    path = ROOT / config["model"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print("Modelo guardado en", path)


if __name__ == "__main__":
    main()

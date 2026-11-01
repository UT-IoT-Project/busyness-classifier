"""Prediccion en vivo.

Cada 'step_seconds' (30 s): pide a la API los ultimos minutos de datos, los
prepara exactamente igual que al entrenar, predice el nivel de ocupacion (0-3)
y lo guarda en el CSV de predicciones.

Ejecutar desde la raiz del repositorio (en Windows):
    set PYTHONPATH=src
    python -m occupancy.predict              con la API real
    python -m occupancy.predict --simulate   con los datos simulados (sin API)
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


def predict_at(now, source, config, bundle):
    """Prediccion para la ventana que termina en 'now'. Devuelve una fila o None."""
    pred_cfg = config["prediction"]
    # Se piden mas minutos que la ventana: la limpieza necesita datos anteriores
    # (suavizado, huecos) para tratar bien el principio de la ventana.
    history = timedelta(minutes=config["windows"]["size_minutes"] + pred_cfg["extra_history_minutes"])
    data = source.fetch(now - history, now)
    if data is None:
        print(format_time(now), "sin datos de la API")
        return None

    tables, pir, labels = clean(data, config["cleaning"])
    df = merge_all(tables, pir, labels)
    windows = build_windows(df, list(tables), config["windows"],
                            config["cleaning"]["interval_seconds"], last_only=True)
    if windows.empty:
        print(format_time(now), "datos insuficientes en la ultima ventana")
        return None

    real_level = windows[LABEL_COLUMN].iloc[0] if LABEL_COLUMN in windows else None
    X = windows.reindex(columns=bundle["features"]).fillna(bundle["fill_values"])

    probabilities = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["model"].classes_)
    level = classes[probabilities.argmax()]
    confidence = probabilities.max()

    # Aviso de poca fiabilidad: el modelo duda, o algun valor no se vio al entrenar
    reliable = confidence >= pred_cfg["min_confidence"]
    if "feature_min" in bundle:
        outside = (X.iloc[0] < bundle["feature_min"]) | (X.iloc[0] > bundle["feature_max"])
        reliable = reliable and not outside.any()

    row = {"timestamp": format_time(windows.index[0]),
           "predicted_level": int(level),
           "confidence": round(float(confidence), 3),
           "reliable": bool(reliable),
           "real_level": "" if real_level is None or real_level != real_level else int(real_level)}
    for cls, p in zip(classes, probabilities):
        row[f"prob_{int(cls)}"] = round(float(p), 3)
    return row


def main():
    parser = argparse.ArgumentParser(description="Prediccion en vivo del nivel de ocupacion")
    parser.add_argument("--simulate", action="store_true",
                        help="usar los datos simulados en lugar de la API (no espera entre pasos)")
    parser.add_argument("--steps", type=int, default=None,
                        help="numero de predicciones antes de parar (por defecto, sin fin)")
    args = parser.parse_args()

    config = load_config()
    bundle = joblib.load(ROOT / config["model"]["path"])
    output = ROOT / config["output"]["predictions_path"]
    step = timedelta(seconds=config["windows"]["step_seconds"])

    if args.simulate:
        source = SimulatedSource(ROOT / "data" / "simulated" / "readings.json")
        first, last = source.time_span()
        now = first + timedelta(minutes=config["windows"]["size_minutes"]
                                + config["prediction"]["extra_history_minutes"])
        output = output.with_name("predictions_simulated.csv")
    else:
        source = ApiSource(config)

    done = 0
    while args.steps is None or done < args.steps:
        if not args.simulate:
            # Hora actual de Amsterdam, sin zona (es como la espera la API)
            now = datetime.now(AMSTERDAM).replace(tzinfo=None)
        elif now > last:
            break

        try:
            row = predict_at(now, source, config, bundle)
        except Exception as error:          # la API puede fallar: se avisa y se sigue
            print(format_time(now), "error:", error)
            row = None

        if row is not None:
            publish_prediction(output, row)
            print(row["timestamp"], "nivel", row["predicted_level"],
                  "confianza", row["confidence"],
                  "" if row["reliable"] else "(poco fiable)",
                  f"real {row['real_level']}" if row["real_level"] != "" else "")

        done += 1
        if args.simulate:
            now += step
        else:
            time.sleep(step.total_seconds())


if __name__ == "__main__":
    main()

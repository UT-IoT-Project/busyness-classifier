"""Prediccion en vivo.

Cada 'step_seconds' (30 s): pide a la API los ultimos minutos de datos, los
prepara exactamente igual que al entrenar, predice el nivel de ocupacion (0-3)
y lo guarda en el CSV de predicciones.

La ventana termina en la hora de la ultima lectura que devuelve la API, no en
la hora del portatil: asi no importa si los relojes del portatil y de la
Raspberry Pi no coinciden.

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


def predict_at(now, source, config, bundle, clock_margin, last_seen=None):
    """Prediccion para la ultima ventana con datos. Devuelve una fila o None.

    Se piden datos alrededor de 'now' con un margen por si los relojes no
    coinciden, y la ventana termina en la ultima lectura recibida.
    Si esa lectura es la misma que la de la prediccion anterior ('last_seen'),
    no hay datos nuevos y no se repite la prediccion.
    """
    pred_cfg = config["prediction"]

    # ================================================================
    # 1. PEDIR LOS DATOS (a la API real o a los datos simulados)
    #    source.fetch esta en api_client.py: hace la peticion GET y
    #    devuelve el campo "data" del JSON, o None si no hay datos (404).
    # ================================================================
    # Se piden mas minutos que la ventana: la limpieza necesita datos anteriores
    # (suavizado, huecos) para tratar bien el principio de la ventana.
    history = timedelta(minutes=config["windows"]["size_minutes"] + pred_cfg["extra_history_minutes"])
    data = source.fetch(now - history - clock_margin, now + clock_margin)
    if data is None:
        print(format_time(now), "sin datos de la API")
        return None

    # ================================================================
    # 2. PASAR LOS DATOS POR LAS ETAPAS (las mismas funciones que en train.py)
    # ================================================================
    # Fase 1, limpieza (cleaning.py): duplicados, rango, picos, huecos y suavizado
    tables, pir, labels = clean(data, config["cleaning"])
    # Fase 2, integracion (integration.py): todo en una tabla, una fila cada 10 s
    df = merge_all(tables, pir, labels)
    # Fase 4, ventanas y caracteristicas (features.py): solo la ultima ventana
    windows = build_windows(df, list(tables), config["windows"],
                            config["cleaning"]["interval_seconds"], last_only=True)

    # Comprobaciones: ventana con pocos datos, o la misma que la prediccion anterior
    if windows.empty:
        print(format_time(now), "datos insuficientes en la ultima ventana")
        return None
    if last_seen is not None and windows.index[0] <= last_seen:
        print(format_time(now), "sin datos nuevos desde", format_time(last_seen))
        return None

    # ================================================================
    # 3. USAR EL MODELO
    # ================================================================
    # Etiqueta real (si la API la trae): solo para comparar, NO entra en el modelo
    real_level = windows[LABEL_COLUMN].iloc[0] if LABEL_COLUMN in windows else None
    # Entrada del modelo: las mismas caracteristicas, en el mismo orden y con el
    # mismo relleno de vacios que al entrenar
    X = windows.reindex(columns=bundle["features"]).fillna(bundle["fill_values"])

    # Aqui se llama al modelo: cada arbol vota y se obtiene la probabilidad de cada nivel
    probabilities = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["model"].classes_)
    level = classes[probabilities.argmax()]
    confidence = probabilities.max()

    # Aviso de poca fiabilidad: el modelo duda, o algun valor no se vio al entrenar
    reliable = confidence >= pred_cfg["min_confidence"]
    if "feature_min" in bundle:
        outside = (X.iloc[0] < bundle["feature_min"]) | (X.iloc[0] > bundle["feature_max"])
        reliable = reliable and not outside.any()

    # ================================================================
    # 4. PREPARAR EL RESULTADO (main lo guarda en el CSV con publish_prediction)
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
    parser = argparse.ArgumentParser(description="Prediccion en vivo del nivel de ocupacion")
    parser.add_argument("--simulate", action="store_true",
                        help="usar los datos simulados en lugar de la API (no espera entre pasos)")
    parser.add_argument("--steps", type=int, default=None,
                        help="numero de predicciones antes de parar (por defecto, sin fin)")
    args = parser.parse_args()

    # ---------- Preparacion, una sola vez al arrancar ----------
    config = load_config()
    # Cargar el modelo entrenado (lo guarda train.py en models/model.joblib)
    bundle = joblib.load(ROOT / config["model"]["path"])
    output = ROOT / config["output"]["predictions_path"]
    step = timedelta(seconds=config["windows"]["step_seconds"])

    # Elegir de donde vienen los datos: simulados (--simulate) o API real
    if args.simulate:
        source = SimulatedSource(ROOT / "data" / "simulated" / "readings.json")
        first, last = source.time_span()
        now = first + timedelta(minutes=config["windows"]["size_minutes"]
                                + config["prediction"]["extra_history_minutes"])
        output = output.with_name("predictions_simulated.csv")
        clock_margin = timedelta(0)        # en la simulacion no hay relojes distintos
    else:
        source = ApiSource(config)
        clock_margin = timedelta(minutes=config["prediction"]["clock_margin_minutes"])

    last_seen = None                       # final de la ultima ventana ya predicha

    # ---------- Ciclo: una prediccion cada 30 s ----------
    done = 0
    while args.steps is None or done < args.steps:
        if not args.simulate:
            # Hora actual de Amsterdam, sin zona (es como la espera la API)
            now = datetime.now(AMSTERDAM).replace(tzinfo=None)
        elif now > last:
            break

        # Pedir datos, pasarlos por las etapas y predecir (todo en predict_at)
        try:
            row = predict_at(now, source, config, bundle, clock_margin, last_seen)
        except Exception as error:          # la API puede fallar: se avisa y se sigue
            print(format_time(now), "error:", error)
            row = None

        # Guardar la prediccion en el CSV (publish.py) y mostrarla en pantalla
        if row is not None:
            last_seen = row["window_end"]
            publish_prediction(output, row)
            print(row["timestamp"], "nivel", row["predicted_level"],
                  "confianza", row["confidence"],
                  "" if row["reliable"] else "(poco fiable)",
                  f"real {row['real_level']}" if row["real_level"] != "" else "")

        # Esperar hasta la siguiente prediccion
        done += 1
        if args.simulate:
            now += step
        else:
            time.sleep(step.total_seconds())


if __name__ == "__main__":
    main()

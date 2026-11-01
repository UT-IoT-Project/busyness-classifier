"""Guardar la prediccion.

Por ahora se guarda en local (decision del grupo): se anade una fila a un CSV,
cuya ruta esta en config.yaml (output.predictions_path). Si mas adelante se
publica por MQTT o por la API, solo cambia esta funcion.
"""
import csv
from pathlib import Path

FIELDS = ["timestamp", "predicted_level", "confidence",
          "prob_0", "prob_1", "prob_2", "prob_3", "reliable", "real_level"]


def publish_prediction(path, row):
    """Anade una fila al CSV de predicciones (crea el archivo y la cabecera si no existen)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})

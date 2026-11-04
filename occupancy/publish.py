"""Save the prediction.

For now it is saved locally (group decision): one row is appended to a CSV
file whose path is in config.yaml (output.predictions_path). If later the
predictions are published via MQTT or the API, only this function changes.
"""
import csv
from pathlib import Path

FIELDS = ["timestamp", "predicted_level", "confidence",
          "prob_0", "prob_1", "prob_2", "prob_3", "reliable", "real_level"]


def publish_prediction(path, row):
    """Append one row to the predictions CSV (creates the file and header if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})

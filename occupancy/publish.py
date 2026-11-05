"""Save the prediction.

One row is appended to a CSV file whose path is in config.yaml
(output.predictions_path). Live predictions are also sent to the API by
predict.py.
"""
import csv
from pathlib import Path

FIELDS = [
    "timestamp", "predicted_level", "random_forest_level",
    "regression_level", "rnn_level", "real_level",
]


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

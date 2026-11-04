"""Phase 2: integration. Joins sensors, PIRs and label into a single table.

This is the code from notebooks/02_integracion.ipynb.
"""
import pandas as pd


def merge_all(tables, pir, labels=None):
    """Join everything returned by cleaning.clean into one table.

    - Index: timestamp (the common time grid from the cleaning).
    - Columns: one per sensor, pir-1, pir-2, pir-3 and, if present, the label.
    labels can be None (live, without a label).
    """
    parts = [table.set_index("timestamp")["value"].rename(sensor)
             for sensor, table in tables.items()]
    if pir is not None and not pir.empty:
        parts.append(pir.set_index("timestamp"))
    if labels is not None:
        parts.append(labels.set_index("timestamp"))
    return pd.concat(parts, axis=1)

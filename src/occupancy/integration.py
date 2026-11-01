"""Fase 2: integracion. Une sensores, PIR y etiqueta en una sola tabla.

Es el codigo del notebook notebooks/02_integracion.ipynb.
"""
import pandas as pd


def merge_all(tables, pir, labels=None):
    """Une todo lo que devuelve cleaning.clean en una tabla.

    - Indice: timestamp (la rejilla comun de la limpieza).
    - Columnas: una por sensor, pir-1, pir-2, pir-3 y, si hay, la etiqueta.
    labels puede ser None (en vivo, sin etiqueta).
    """
    parts = [table.set_index("timestamp")["value"].rename(sensor)
             for sensor, table in tables.items()]
    if pir is not None and not pir.empty:
        parts.append(pir.set_index("timestamp"))
    if labels is not None:
        parts.append(labels.set_index("timestamp"))
    return pd.concat(parts, axis=1)

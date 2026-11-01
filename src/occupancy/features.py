"""Fase 4: transformacion. Convierte la tabla integrada en una fila por ventana.

Es el codigo del notebook notebooks/03_caracteristicas.ipynb.
Las ventanas solo miran hacia atras, asi que funcionan igual en vivo.

Tolerancia a fallos de sensores:
- Si un sensor no tiene datos, sus caracteristicas quedan vacias (NaN) y la
  ventana sigue siendo valida mientras la mayoria de sensores si tengan datos.
  Al predecir, los vacios se rellenan y la prediccion se marca como poco fiable.
- Los PIR se buscan por nombre (todas las columnas 'pir-<ID>'), sean cuantos sean.
"""
import numpy as np
import pandas as pd

LABEL_COLUMN = "label"


def most_frequent(values):
    """Valor mas frecuente de una ventana, sin contar los vacios."""
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def pir_columns(df):
    """Columnas de los PIR que haya en la tabla (pir-1, pir-2...)."""
    return [column for column in df.columns if str(column).startswith("pir-")]


def build_windows(df, sensors, window_cfg, interval_seconds,
                  label_column="busy_level", last_only=False):
    """Devuelve una tabla con una fila por ventana valida.

    - Una ventana de 'size_minutes' cada 'step_seconds'.
    - Con last_only=True devuelve solo la ultima ventana (prediccion en vivo);
      si no es valida, devuelve una tabla vacia.
    - Columnas: las caracteristicas y, si df tiene etiqueta, 'label'.
    - Una ventana es valida si es completa, si al menos 'min_sensor_fraction'
      de los sensores tienen 'min_coverage' de datos y, al entrenar, si tiene etiqueta.
    """
    window = f"{window_cfg['size_minutes']}min"
    step = window_cfg["step_seconds"] // interval_seconds
    min_coverage = window_cfg.get("min_coverage", 0.8)
    min_sensor_fraction = window_cfg.get("min_sensor_fraction", 0.5)

    # Columnas que falten (sensor sin datos) se crean vacias
    df = df.reindex(columns=list(dict.fromkeys(list(df.columns) + list(sensors))))
    roll = df.rolling(window, min_periods=1)

    # Caracteristicas
    features = pd.DataFrame(index=df.index)
    for sensor in sensors:
        features[f"{sensor}_mean"] = roll[sensor].mean()

    features["co2Equivalent_change"] = (
        df["co2Equivalent"] - df["co2Equivalent"].shift(freq=window)).reindex(df.index)
    features["mic_vol_max"] = roll["mic_vol"].max()
    features["mic_vol_std"] = roll["mic_vol"].std()

    pirs = pir_columns(df)
    if pirs:
        features["pir_mean"] = df[pirs].mean(axis=1).rolling(window).mean()
        features["pir_max"] = df[pirs].max(axis=1).rolling(window).max()
    else:
        features["pir_mean"] = np.nan
        features["pir_max"] = np.nan

    # Etiqueta de cada ventana: el nivel mas frecuente
    has_label = label_column in df.columns and df[label_column].notna().any()
    if has_label:
        features[LABEL_COLUMN] = df[label_column].rolling(window).apply(
            most_frequent, raw=True)

    # Cobertura: parte de la ventana con datos, sensor a sensor
    sensor_coverage = df[sensors].notna().rolling(window).mean()
    enough_sensors = (sensor_coverage >= min_coverage).mean(axis=1) >= min_sensor_fraction

    # Una ventana cada 'step_seconds' (o solo la ultima, en vivo)
    if last_only:
        windows = features.iloc[[-1]]
        enough_sensors = enough_sensors.iloc[[-1]]
    else:
        windows = features.iloc[::step]
        enough_sensors = enough_sensors.iloc[::step]

    valid = (windows.index >= df.index[0] + pd.Timedelta(window)) & enough_sensors
    if has_label and not last_only:
        valid = valid & windows[LABEL_COLUMN].notna()

    windows = windows[valid].copy()
    if has_label and not last_only:
        windows[LABEL_COLUMN] = windows[LABEL_COLUMN].astype(int)
    return windows

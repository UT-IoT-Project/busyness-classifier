"""Fase 4: transformacion. Convierte la tabla integrada en una fila por ventana.

Es el codigo del notebook notebooks/03_caracteristicas.ipynb.
Las ventanas solo miran hacia atras, asi que funcionan igual en vivo.
"""
import numpy as np
import pandas as pd

PIR_COLUMNS = ["pir-1", "pir-2", "pir-3"]
LABEL_COLUMN = "label"


def most_frequent(values):
    """Valor mas frecuente de una ventana, sin contar los vacios."""
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def build_windows(df, sensors, window_cfg, interval_seconds,
                  label_column="busynessLabel", min_coverage=0.8, last_only=False):
    """Devuelve una tabla con una fila por ventana valida.

    - Una ventana de 'size_minutes' cada 'step_seconds'.
    - Con last_only=True devuelve solo la ultima ventana (prediccion en vivo);
      si no es valida, devuelve una tabla vacia.
    - Columnas: las caracteristicas y, si df tiene etiqueta, 'label'.
    - Solo ventanas completas, con al menos 'min_coverage' de datos y,
      si hay etiqueta, con etiqueta.
    """
    window = f"{window_cfg['size_minutes']}min"
    step = window_cfg["step_seconds"] // interval_seconds
    roll = df.rolling(window, min_periods=1)

    features = pd.DataFrame(index=df.index)
    for sensor in sensors:
        features[f"{sensor}_mean"] = roll[sensor].mean()

    features["co2Equivalent_change"] = (
        df["co2Equivalent"] - df["co2Equivalent"].shift(freq=window))
    features["mic_vol_max"] = roll["mic_vol"].max()
    features["mic_vol_std"] = roll["mic_vol"].std()
    pir_activity = df[PIR_COLUMNS].mean(axis=1)
    features["pir_mean"] = pir_activity.rolling(window).mean()
    features["pir_max"] = df[PIR_COLUMNS].max(axis=1).rolling(window).max()

    has_label = label_column in df.columns
    if has_label:
        features[LABEL_COLUMN] = df[label_column].rolling(window).apply(
            most_frequent, raw=True)

    coverage = df[sensors].notna().all(axis=1).rolling(window).mean()
    if last_only:
        windows = features.iloc[[-1]]
        coverage = coverage.iloc[[-1]]
    else:
        windows = features.iloc[::step]
        coverage = coverage.iloc[::step]
    valid = (windows.index >= df.index[0] + pd.Timedelta(window)) & (coverage >= min_coverage)
    if has_label and not last_only:
        valid = valid & windows[LABEL_COLUMN].notna()

    windows = windows[valid].copy()
    if has_label and not last_only:
        windows[LABEL_COLUMN] = windows[LABEL_COLUMN].astype(int)
    return windows

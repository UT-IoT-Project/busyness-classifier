"""Fase 1: limpieza de datos.

Es el codigo del notebook notebooks/01_limpieza.ipynb, pasado a funciones para
que el entrenamiento y la prediccion en vivo limpien exactamente igual.
Todos los ajustes se leen de la seccion "cleaning" de config.yaml.

Orden de las etapas (ver el documento del pipeline):
    0. Convertir la respuesta de la API en tablas
    1.1 Duplicados
    1.2 Control de rango
    1.3 Picos aislados
    1.4 Huecos (rejilla comun de tiempo + interpolar huecos cortos)
    1.5 Suavizado (opcional)

Uso:
    from occupancy.cleaning import clean
    tables, pir = clean(data, config["cleaning"])
"""
import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
SKIPPED_SENSORS = ("PIR", "busy_level")


# ---------- 0. De la respuesta de la API a tablas ----------

def readings_to_tables(data):
    """Convierte el campo 'data' de la API en tablas de pandas.

    Devuelve:
    - tables: diccionario {sensor: tabla con columnas timestamp y value}
    - pir: una tabla con timestamp y una columna por PIR (pir-1, pir-2...)
    PIR y busy_level no van en tables.
    """
    tables = {}
    for sensor, readings in data.items():
        if sensor in SKIPPED_SENSORS:
            continue
        table = pd.DataFrame(readings, columns=["timestamp", "value"])
        table["timestamp"] = pd.to_datetime(table["timestamp"], format=TIME_FORMAT)
        tables[sensor] = table

    rows = []
    for reading in data.get("PIR", []):
        row = {"timestamp": reading["timestamp"]}
        for p in reading["values"]:
            row[p["ID"]] = p["value"]
        rows.append(row)
    pir = pd.DataFrame(rows)
    if not pir.empty:
        pir["timestamp"] = pd.to_datetime(pir["timestamp"], format=TIME_FORMAT)

    return tables, pir


# ---------- 1.1 Duplicados ----------

def remove_duplicates(table):
    """Quita las filas exactamente iguales (QoS 1 puede entregar un mensaje dos veces).

    Compara la fila completa, no solo la hora.
    """
    return table.drop_duplicates()


# ---------- 1.2 Control de rango ----------

def check_range(table, limits):
    """Vacia (NaN) los valores fuera de [min, max], sin borrar la fila."""
    table = table.copy()
    out_of_range = (table["value"] < limits["min"]) | (table["value"] > limits["max"])
    table["value"] = table["value"].mask(out_of_range)
    return table


# ---------- 1.3 Picos aislados ----------

def remove_spikes(table, threshold, window_readings):
    """Vacia los valores que se alejan de la mediana de sus vecinos mas que el umbral."""
    table = table.copy()
    median = table["value"].rolling(window_readings, center=True, min_periods=1).median()
    distance = table["value"].sub(median).abs()
    table["value"] = table["value"].mask(distance > threshold)
    return table


# ---------- 1.4 Huecos ----------

def to_grid(table, interval):
    """Pasa una tabla a la rejilla regular de tiempo.

    Devuelve una tabla con timestamp como indice y una fila exacta cada 'interval'.
    Los momentos sin lectura quedan como NaN. Sirve tambien para la tabla de PIR,
    que tiene varias columnas.
    """
    return table.set_index("timestamp").resample(interval).mean()


def fill_short_gaps(series, max_gap_readings):
    """Interpola solo los huecos de como mucho max_gap_readings lecturas seguidas.

    Los huecos mas largos se dejan enteros vacios (no se rellena solo su principio).
    """
    empty = series.isna()
    group = (empty != empty.shift()).cumsum()
    gap_length = empty.groupby(group).transform("sum")
    short_gap = empty & (gap_length <= max_gap_readings)
    filled = series.interpolate(method="time", limit_area="inside")
    return series.mask(short_gap, filled)


# ---------- 1.5 Suavizado ----------

def smooth(series, window):
    """Media movil que solo mira hacia atras; no rellena los huecos que ya estaban vacios."""
    smoothed = series.rolling(window, min_periods=1).mean()
    return smoothed.where(series.notna())


# ---------- Todo el proceso ----------

def clean(data, cfg):
    """Aplica toda la limpieza al campo 'data' de la API.

    cfg es la seccion "cleaning" de config.yaml.
    Devuelve (tables, pir), todas en la misma rejilla de tiempo.
    """
    tables, pir = readings_to_tables(data)
    interval = pd.Timedelta(seconds=cfg["interval_seconds"])
    max_gap = cfg["max_gap_readings"]
    smoothing = cfg["smoothing"]
    smooth_window = pd.Timedelta(seconds=smoothing["window_seconds"])

    for sensor in list(tables):
        table = remove_duplicates(tables[sensor])
        if sensor in cfg["limits"]:
            table = check_range(table, cfg["limits"][sensor])
        if sensor in cfg["spike_thresholds"]:
            table = remove_spikes(table, cfg["spike_thresholds"][sensor],
                                  cfg["spike_window_readings"])

        series = to_grid(table, interval)["value"]
        series = fill_short_gaps(series, max_gap)
        if smoothing["enabled"] and sensor in smoothing["sensors"]:
            series = smooth(series, smooth_window)

        tables[sensor] = series.reset_index(name="value")

    if not pir.empty:
        grid = to_grid(remove_duplicates(pir), interval)
        for column in grid.columns:
            grid[column] = fill_short_gaps(grid[column], max_gap)
        pir = grid.reset_index()

    return tables, pir

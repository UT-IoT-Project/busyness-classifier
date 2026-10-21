"""Fase 4: transformacion. Convierte la tabla en una fila por ventana de tiempo."""


def make_windows(table, size_minutes, step_seconds):
    """Divide la tabla en ventanas solapadas y devuelve una lista de trozos."""
    raise NotImplementedError


def window_features(window):
    """Calcula las caracteristicas de una ventana y las devuelve como diccionario.

    Ideas (por decidir): pendiente y nivel relativo del CO2 equivalente,
    media, maximo y suma de los PIR, nivel de sonido relativo al fondo.
    """
    raise NotImplementedError

"""Fase 1: limpieza. Ver la tabla de limpieza del documento del pipeline."""


def remove_duplicates(df):
    """Quita lecturas repetidas (QoS 1 puede entregar el mismo mensaje dos veces)."""
    raise NotImplementedError


def check_range(df, sensor):
    """Descarta valores fuera de lo que el sensor puede medir y los picos aislados."""
    raise NotImplementedError


def fill_gaps(df, max_gap_seconds):
    """Interpola los huecos cortos y marca como invalidos los huecos largos."""
    raise NotImplementedError


def smooth(df, minutes):
    """Media movil que solo mira hacia atras (igual al entrenar y al predecir)."""
    raise NotImplementedError

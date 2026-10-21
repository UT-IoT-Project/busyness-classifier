"""Validacion y metricas."""


def leave_one_group_out(features, labels, groups):
    """Entrena dejando fuera una sala o sesion completa cada vez."""
    raise NotImplementedError


def metrics(y_true, y_pred):
    """Devuelve MAE, RMSE y el porcentaje de errores de como maximo 1 y 2 personas."""
    raise NotImplementedError

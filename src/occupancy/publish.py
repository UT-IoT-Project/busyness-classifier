"""Guardar la prediccion.

Por ahora se guarda en local (decision del grupo): se anade una fila a un CSV,
cuya ruta esta en config.yaml (output.predictions_path). Si mas adelante se
publica por MQTT o por la API, solo cambia esta funcion.
"""


def publish_prediction(config, timestamp, level, probabilities):
    """Anade al CSV de predicciones la hora, el nivel predicho (0-3) y la
    probabilidad de cada nivel.

    TODO: crear la carpeta si no existe, escribir la cabecera la primera vez
    y anadir una fila por prediccion.
    """
    raise NotImplementedError

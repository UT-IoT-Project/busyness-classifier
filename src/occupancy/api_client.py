"""Pide lecturas a la API del backend (repositorio platform).

Contrato: GET /api/v1/readings/{building}/{roomNumber}
          ?start_time=...&end_time=...&sensors=...&sensors=...
Las fechas van en formato 'YYYY-MM-DD HH:mm:ss.SSS', hora local del servidor.
Respuesta: {"status", "message", "data": {sensor: [lecturas]}}
"""


def fetch_readings(config, start_time, end_time):
    """Devuelve las lecturas de todos los sensores de la sala entre dos fechas.

    TODO:
    - Construir la URL con building y room de la configuracion.
    - Pasar start_time, end_time y un parametro 'sensors' por cada sensor.
    - Comprobar el campo 'status' y tratar los errores 400, 404 y 500.
    - Devolver un diccionario {nombre_sensor: DataFrame con columnas timestamp y value}.
    - Los PIR llegan agrupados: {timestamp, values: [{ID, value}]}; pasarlos a filas.
    """
    raise NotImplementedError

"""Pedir lecturas: a la API real del repositorio platform o a los datos simulados.

Contrato (platform/docs/api/openapi.yaml, version del 30 de octubre):
    GET /api/v1/readings/{building}/{roomNumber}
        ?start_time=...&end_time=...&sensors=...&sensors=...
- Fechas 'YYYY-MM-DD HH:mm:ss.SSS' en hora local de Europe/Amsterdam.
- Rango de 0 a 24 horas como maximo.
- 404 = ningun sensor pedido tiene datos en ese rango (no es un error grave).
"""
import json
from datetime import datetime
from urllib.parse import quote, urlencode

import requests

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def format_time(moment):
    """Fecha -> texto con el formato de la API (milisegundos)."""
    return moment.strftime(TIME_FORMAT)[:-3]


class ApiSource:
    """Lee de la API real."""

    def __init__(self, config, timeout=10):
        api = config["api"]
        self.url = f"{api['base_url'].rstrip('/')}/api/v1/readings/{api['building']}/{api['room']}"
        self.sensors = config["sensors"]
        self.timeout = timeout

    def fetch(self, start, end):
        """Devuelve el campo 'data' de la respuesta, o None si no hay datos (404)."""
        params = {"start_time": format_time(start),
                  "end_time": format_time(end),
                  "sensors": self.sensors}
        # doseq repite la clave por cada sensor; quote codifica el espacio como %20
        query = urlencode(params, doseq=True, quote_via=quote)
        response = requests.get(f"{self.url}?{query}", timeout=self.timeout)
        if response.status_code == 404:
            return None
        body = response.json()
        if response.status_code != 200:
            raise RuntimeError(f"La API respondio {response.status_code}: {body.get('message')}")
        return body["data"]


class SimulatedSource:
    """Lee de data/simulated/readings.json como si fuera la API.

    Sirve para probar la prediccion en vivo sin la API: devuelve solo las
    lecturas entre start y end, con el mismo formato que la API real.
    """

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)["data"]

    def time_span(self):
        """Primera y ultima hora de los datos simulados."""
        stamps = [datetime.strptime(r["timestamp"], TIME_FORMAT)
                  for readings in self.data.values() for r in readings]
        return min(stamps), max(stamps)

    def fetch(self, start, end):
        result = {}
        for sensor, readings in self.data.items():
            result[sensor] = [r for r in readings
                              if start <= datetime.strptime(r["timestamp"], TIME_FORMAT) <= end]
        if not any(result.values()):
            return None
        return result

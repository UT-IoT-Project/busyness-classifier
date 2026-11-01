"""Descargar una sesion de recogida de datos reales de la API.

Una sesion es un periodo en el que los nodos enviaban datos y una persona
marcaba el nivel de ocupacion (busy_level). Cada sesion se guarda en un
archivo propio en data/raw/<nombre>.json, con el mismo formato que la API.
train.py usa cada archivo como un grupo para validar: entrena con unas
sesiones y prueba con otra.

Ejecutar desde la raiz del repositorio (en Windows):
    set PYTHONPATH=src
    python -m occupancy.download --session sesion1 --start "2026-11-03 10:00" --end "2026-11-03 12:30"

Las horas van en hora local de Amsterdam, como las pide la API.
"""
import argparse
import json
from datetime import datetime, timedelta

from occupancy.api_client import ApiSource, format_time
from occupancy.train import ROOT, load_config

MAX_CHUNK = timedelta(hours=24)        # la API no acepta mas de 24 h por peticion
ONE_MS = timedelta(milliseconds=1)


def download(source, start, end):
    """Pide el periodo en trozos de 24 h como maximo y une las lecturas."""
    merged = {}
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + MAX_CHUNK, end)
        print("Pidiendo", format_time(chunk_start), "->", format_time(chunk_end))
        data = source.fetch(chunk_start, chunk_end)
        if data is None:
            print("  sin datos en este trozo")
        else:
            for sensor, readings in data.items():
                merged.setdefault(sensor, []).extend(readings)
        # Los extremos se incluyen: el siguiente trozo empieza 1 ms despues
        chunk_start = chunk_end + ONE_MS
    return merged


def main():
    parser = argparse.ArgumentParser(description="Descargar una sesion de datos reales")
    parser.add_argument("--session", required=True, help="nombre de la sesion (nombre del archivo)")
    parser.add_argument("--start", required=True, help='inicio, "YYYY-MM-DD HH:MM" (hora de Amsterdam)')
    parser.add_argument("--end", required=True, help='fin, "YYYY-MM-DD HH:MM" (hora de Amsterdam)')
    args = parser.parse_args()

    config = load_config()
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    if end <= start:
        raise SystemExit("El fin tiene que ser posterior al inicio")

    data = download(ApiSource(config), start, end)
    if not any(data.values()):
        raise SystemExit("La API no devolvio datos en ese periodo: no se guarda nada")

    labels = data.get(config["cleaning"]["label"]["sensor"], [])
    print({sensor: len(readings) for sensor, readings in data.items()})
    if not labels:
        print("AVISO: la sesion no tiene etiqueta (busy_level): no servira para entrenar")

    path = ROOT / "data" / "raw" / f"{args.session}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    session = {"session": args.session,
               "building": config["api"]["building"],
               "room": config["api"]["room"],
               "start": format_time(start),
               "end": format_time(end),
               "data": data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f)
    print("Guardado en", path)


if __name__ == "__main__":
    main()

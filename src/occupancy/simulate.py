"""Genera datos simulados con el mismo formato que la respuesta de la API.

Sirven para probar el pipeline antes de que la API funcione. Se guardan en
data/simulated/ y nunca deben mezclarse con datos reales.

Ejecutar desde la raiz del repositorio (en Windows):
    set PYTHONPATH=src
    python -m occupancy.simulate

Archivos que genera:
    readings.json  respuesta completa de la API, con fallos metidos a proposito
    labels.csv     numero real de personas en cada instante (sin fallos)
    faults.json    lista de los fallos introducidos, para comprobar la limpieza

IMPORTANTE: la relacion entre personas y sensores es inventada. Estos datos
sirven para comprobar que el codigo funciona, no para juzgar si el modelo es bueno.
"""
import copy
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# Parametros de la simulacion
START_TIME = "2026-10-20 09:00:00.000"
MINUTES = 180            # duracion total
INTERVAL_SECONDS = 10    # cada cuanto "publica" el nodo (supuesto, por confirmar)
SEED = 42                # semilla fija: siempre salen los mismos datos

# Carpeta de salida: <raiz del repositorio>/data/simulated
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "simulated"

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
PIR_IDS = ["pir-1", "pir-2", "pir-3"]
SCALAR_SENSORS = ["pressure", "hum", "temp", "co2equivalent",
                  "voc", "airresistance", "mic_vol"]


def format_time(moment):
    """Convierte una fecha al formato de la API: 'YYYY-MM-DD HH:mm:ss.SSS'."""
    return moment.strftime(TIME_FORMAT)[:-3]


def simulate_people(steps, interval_seconds, rng):
    """Numero de personas en cada paso: bloques de 10 a 30 minutos con un valor fijo."""
    people = []
    while len(people) < steps:
        count = rng.choice([0, 0, 2, 5, 8, 12, 15])
        block_minutes = rng.randint(10, 30)
        people.extend([count] * (block_minutes * 60 // interval_seconds))
    return people[:steps]


def simulate_readings(start_time, minutes, interval_seconds, rng):
    """Simula los mensajes que llegarian del nodo, uno por instante.

    Devuelve dos listas:
    - messages: cada mensaje tiene 'timestamp', los sensores y 'PIRs'
      (como el JSON del ESP32 con la hora que anade Mosquitto).
    - labels: el numero real de personas en cada instante.
    """
    start = datetime.strptime(start_time, TIME_FORMAT)
    steps = minutes * 60 // interval_seconds
    people = simulate_people(steps, interval_seconds, rng)

    # Valores iniciales de los sensores que cambian despacio
    co2, temp, hum = 450.0, 21.0, 45.0
    messages, labels = [], []

    for i in range(steps):
        timestamp = format_time(start + timedelta(seconds=i * interval_seconds))
        n = people[i]

        # Cada sensor se acerca poco a poco a un nivel que depende de la gente
        co2 += (450 + 40 * n - co2) * 0.02 + rng.gauss(0, 5)
        temp += (21 + 0.1 * n - temp) * 0.01 + rng.gauss(0, 0.02)
        hum += (45 + 0.5 * n - hum) * 0.01 + rng.gauss(0, 0.1)

        # Probabilidad de que un PIR detecte movimiento: crece con la gente
        p_active = 1 - math.exp(-0.3 * n)

        messages.append({
            "timestamp": timestamp,
            "pressure": round(1013 + rng.gauss(0, 0.3), 2),
            "hum": round(hum, 2),
            "temp": round(temp, 2),
            "co2equivalent": round(co2, 1),
            "voc": round(max(0.0, 0.3 + 0.03 * n + rng.gauss(0, 0.05)), 3),
            "airresistance": round(50000 - 1500 * n + rng.gauss(0, 500), 1),
            "mic_vol": round(min(1.0, max(0.0, 0.05 + 0.02 * n + rng.gauss(0, 0.03))), 3),
            "PIRs": [{"id": pir_id, "value": 1 if rng.random() < p_active else 0}
                     for pir_id in PIR_IDS],
        })
        labels.append({"timestamp": timestamp, "people": n})

    return messages, labels


def add_faults(messages, interval_seconds, rng):
    """Introduce a proposito los fallos que la limpieza debe arreglar.

    Devuelve los mensajes con fallos y una lista que describe cada fallo.
    """
    messages = copy.deepcopy(messages)
    faults = []

    # 1. Hueco largo: el nodo deja de enviar durante 3 minutos
    gap_length = 3 * 60 // interval_seconds
    start = rng.randint(100, len(messages) - gap_length - 100)
    faults.append({"type": "long_gap", "from": messages[start]["timestamp"],
                   "to": messages[start + gap_length - 1]["timestamp"]})
    del messages[start:start + gap_length]

    # 2. Huecos cortos: se pierden mensajes sueltos
    for index in sorted(rng.sample(range(10, len(messages) - 10), 3), reverse=True):
        faults.append({"type": "short_gap", "timestamp": messages[index]["timestamp"]})
        del messages[index]

    # 3. CO2 ausente: el campo es opcional y a veces no llega
    for index in rng.sample(range(len(messages)), len(messages) // 50):
        del messages[index]["co2equivalent"]
        faults.append({"type": "missing_co2", "timestamp": messages[index]["timestamp"]})

    # 4. Valores imposibles en un sensor concreto
    impossible = [("co2equivalent", -50.0), ("co2equivalent", -50.0),
                  ("temp", 150.0), ("hum", 130.0)]
    for sensor, value in impossible:
        index = rng.randrange(len(messages))
        messages[index][sensor] = value
        faults.append({"type": "impossible_value", "sensor": sensor,
                       "timestamp": messages[index]["timestamp"]})

    # 5. Picos aislados en el microfono
    for index in rng.sample(range(1, len(messages) - 1), 4):
        messages[index]["mic_vol"] = 1.0
        faults.append({"type": "spike", "sensor": "mic_vol",
                       "timestamp": messages[index]["timestamp"]})

    # 6. Duplicados: el mismo mensaje llega dos veces (QoS 1)
    for index in sorted(rng.sample(range(len(messages)), 8), reverse=True):
        messages.insert(index + 1, copy.deepcopy(messages[index]))
        faults.append({"type": "duplicate", "timestamp": messages[index]["timestamp"]})

    return messages, faults


def to_api_response(messages):
    """Convierte los mensajes al formato de respuesta de la API (openapi.yaml)."""
    data = {"PIR": []}
    for sensor in SCALAR_SENSORS:
        data[sensor] = []

    for message in messages:
        timestamp = message["timestamp"]
        for sensor in SCALAR_SENSORS:
            if sensor in message:
                data[sensor].append({"timestamp": timestamp, "value": message[sensor]})
        data["PIR"].append({
            "timestamp": timestamp,
            "values": [{"ID": pir["id"], "value": pir["value"]} for pir in message["PIRs"]],
        })

    # Aun no sabemos que mide busy_level: se deja vacio, como permite la API
    data["busy_level"] = []

    return {"status": 200, "message": "data retrieved successfully", "data": data}


def main():
    """Genera los datos simulados y los guarda en data/simulated/."""
    rng = random.Random(SEED)
    messages, labels = simulate_readings(START_TIME, MINUTES, INTERVAL_SECONDS, rng)
    faulty_messages, faults = add_faults(messages, INTERVAL_SECONDS, rng)
    response = to_api_response(faulty_messages)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "readings.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=1)

    with open(OUTPUT_DIR / "labels.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "people"])
        writer.writeheader()
        writer.writerows(labels)

    with open(OUTPUT_DIR / "faults.json", "w", encoding="utf-8") as f:
        json.dump(faults, f, indent=1)

    print(f"{len(messages)} mensajes simulados, {len(faults)} fallos introducidos")
    print(f"Guardado en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

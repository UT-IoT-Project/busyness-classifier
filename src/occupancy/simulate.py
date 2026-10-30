"""Genera datos simulados con el mismo formato que la respuesta de la API.

Sirven para probar el pipeline antes de tener datos reales. Se guardan en
data/simulated/ y nunca deben mezclarse con datos reales.
Formato: contrato de platform/docs/api/openapi.yaml (version del 30 de octubre).

Ejecutar desde la raiz del repositorio (en Windows):
    set PYTHONPATH=src
    python -m occupancy.simulate

Archivos que genera:
    readings.json  respuesta completa de la API, con fallos metidos a proposito
    labels.csv     numero real de personas y nivel de ocupacion (0-3) en cada instante
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
PIR_IDS = [1, 2, 3]
PIR_POLLS = 10           # el nodo lee cada PIR 10 veces y envia el % de actividad
LABEL = "busynessLabel"
SCALAR_SENSORS = ["pressure", "humidity", "temperature", "co2Equivalent", "voc",
                  "iaq", "mic_vol", "wifiPackets", "bleAdverts", LABEL]


def format_time(moment):
    """Convierte una fecha al formato de la API: 'YYYY-MM-DD HH:mm:ss.SSS'."""
    return moment.strftime(TIME_FORMAT)[:-3]


def busy_level_from_people(n):
    """Nivel de ocupacion (0-3) que pondria la persona que etiqueta.

    Criterio provisional, por acordar con el grupo:
    0 = vacia, 1 = de 1 a 5 personas, 2 = de 6 a 10, 3 = mas de 10.
    """
    if n == 0:
        return 0
    if n <= 5:
        return 1
    if n <= 10:
        return 2
    return 3


def simulate_people(steps, interval_seconds, rng):
    """Numero de personas en cada paso: bloques de 10 a 30 minutos con un valor fijo."""
    people = []
    while len(people) < steps:
        count = rng.choice([0, 0, 3, 5, 8, 12, 15])
        block_minutes = rng.randint(10, 30)
        people.extend([count] * (block_minutes * 60 // interval_seconds))
    return people[:steps]


def noisy_count(mean, rng):
    """Un conteo entero, nunca negativo, alrededor de 'mean'."""
    return max(0, round(rng.gauss(mean, math.sqrt(mean) + 1)))


def simulate_readings(start_time, minutes, interval_seconds, rng):
    """Simula los mensajes que llegarian del nodo, uno por instante.

    Devuelve dos listas:
    - messages: cada mensaje tiene 'timestamp', los sensores, la etiqueta
      (busynessLabel, de 0 a 3, la pone una persona) y 'pirs'.
    - labels: el numero real de personas y el nivel en cada instante.
    """
    start = datetime.strptime(start_time, TIME_FORMAT)
    steps = minutes * 60 // interval_seconds
    people = simulate_people(steps, interval_seconds, rng)

    # Valores iniciales de los sensores que cambian despacio
    co2, temperature, humidity = 450.0, 21.0, 45.0
    messages, labels = [], []

    for i in range(steps):
        timestamp = format_time(start + timedelta(seconds=i * interval_seconds))
        n = people[i]

        # Cada sensor se acerca poco a poco a un nivel que depende de la gente
        co2 += (450 + 40 * n - co2) * 0.02 + rng.gauss(0, 5)
        temperature += (21 + 0.1 * n - temperature) * 0.01 + rng.gauss(0, 0.02)
        humidity += (45 + 0.5 * n - humidity) * 0.01 + rng.gauss(0, 0.1)

        # Probabilidad de que un PIR detecte movimiento en una lectura
        p_active = 1 - math.exp(-0.3 * n)

        messages.append({
            "timestamp": timestamp,
            "pressure": round(1013 + rng.gauss(0, 0.3), 2),
            "humidity": round(humidity, 2),
            "temperature": round(temperature, 2),
            "co2Equivalent": round(co2, 1),
            "voc": round(max(0.0, 0.3 + 0.03 * n + rng.gauss(0, 0.05)), 3),
            "iaq": round(min(500.0, max(0.0, 25 + 12 * n + rng.gauss(0, 8))), 1),
            "mic_vol": round(min(1.0, max(0.0, 0.05 + 0.02 * n + rng.gauss(0, 0.03))), 3),
            "wifiPackets": noisy_count(8 + 6 * n, rng),
            "bleAdverts": noisy_count(4 + 3 * n, rng),
            LABEL: busy_level_from_people(n),
            "pirs": [{"ID": pir_id,
                      "activity": 100 * sum(rng.random() < p_active
                                            for _ in range(PIR_POLLS)) // PIR_POLLS}
                     for pir_id in PIR_IDS],
        })
        labels.append({"timestamp": timestamp, "people": n,
                       LABEL: busy_level_from_people(n)})

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

    # 2. Huecos cortos: se pierden mensajes sueltos (con QoS 0 no se reenvian)
    for index in sorted(rng.sample(range(10, len(messages) - 10), 3), reverse=True):
        faults.append({"type": "short_gap", "timestamp": messages[index]["timestamp"]})
        del messages[index]

    # 3. Valores imposibles en un sensor concreto
    impossible = [("co2Equivalent", -50.0), ("co2Equivalent", -50.0),
                  ("temperature", 150.0), ("humidity", 130.0)]
    for sensor, value in impossible:
        index = rng.randrange(len(messages))
        messages[index][sensor] = value
        faults.append({"type": "impossible_value", "sensor": sensor,
                       "timestamp": messages[index]["timestamp"]})

    # 4. Picos aislados en el microfono
    for index in rng.sample(range(1, len(messages) - 1), 4):
        messages[index]["mic_vol"] = 1.0
        faults.append({"type": "spike", "sensor": "mic_vol",
                       "timestamp": messages[index]["timestamp"]})

    # 5. Duplicados: el mismo mensaje llega dos veces (poco probable con QoS 0,
    #    pero posible si alguna vez se publica con QoS 1)
    for index in sorted(rng.sample(range(len(messages)), 8), reverse=True):
        messages.insert(index + 1, copy.deepcopy(messages[index]))
        faults.append({"type": "duplicate", "timestamp": messages[index]["timestamp"]})

    return messages, faults


def to_api_response(messages):
    """Convierte los mensajes al formato de respuesta de la API (openapi.yaml)."""
    data = {"pirs": []}
    for sensor in SCALAR_SENSORS:
        data[sensor] = []

    for message in messages:
        timestamp = message["timestamp"]
        for sensor in SCALAR_SENSORS:
            if sensor in message:
                data[sensor].append({"timestamp": timestamp, "value": message[sensor]})
        data["pirs"].append({"timestamp": timestamp, "values": message["pirs"]})

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
        writer = csv.DictWriter(f, fieldnames=["timestamp", "people", LABEL])
        writer.writeheader()
        writer.writerows(labels)

    with open(OUTPUT_DIR / "faults.json", "w", encoding="utf-8") as f:
        json.dump(faults, f, indent=1)

    print(f"{len(messages)} mensajes simulados, {len(faults)} fallos introducidos")
    print(f"Guardado en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

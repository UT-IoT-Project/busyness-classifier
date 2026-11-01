"""Generates simulated data in the same format as the API response.

Used to test the pipeline before there is real data. Saved in data/simulated/
and must never be mixed with real data.
Format: contract in platform/docs/api/openapi.yaml.

Run from the repository root (Windows PowerShell):
    $env:PYTHONPATH = "src"
    python -m occupancy.simulate

Files it generates:
    readings.json  full API response, with faults added on purpose
    labels.csv     true number of people and occupancy level (0-3) at each moment
    faults.json    list of the faults added, to check the cleaning

IMPORTANT: the relation between people and sensors is made up. This data is
for checking that the code works, not for judging whether the model is good.
"""
import copy
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# Simulation parameters
START_TIME = "2026-10-20 09:00:00.000"
MINUTES = 180            # total duration
INTERVAL_SECONDS = 10    # how often the node "publishes" (assumption, to be confirmed)
SEED = 42                # fixed seed: always the same data

# Output folder: <repository root>/data/simulated
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "simulated"

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
PIR_IDS = [1, 2, 3]
PIR_POLLS = 10           # the node reads each PIR 10 times and sends the % of activity
LABEL = "busy_level"
SCALAR_SENSORS = ["pressure", "humidity", "temperature", "co2Equivalent", "voc",
                  "iaq", "mic_vol", "wifiPackets", "bleAdverts", LABEL]


def format_time(moment):
    """Convert a datetime to the API format: 'YYYY-MM-DD HH:mm:ss.SSS'."""
    return moment.strftime(TIME_FORMAT)[:-3]


def busy_level_from_people(n):
    """Occupancy level (0-3) the labelling person would set.

    Rough mapping used only for the simulation:
    0 = empty, 1 = 1 to 5 people, 2 = 6 to 10, 3 = more than 10.
    """
    if n == 0:
        return 0
    if n <= 5:
        return 1
    if n <= 10:
        return 2
    return 3


def simulate_people(steps, interval_seconds, rng):
    """Number of people at each step: blocks of 10 to 30 minutes with a fixed value."""
    people = []
    while len(people) < steps:
        count = rng.choice([0, 0, 3, 5, 8, 12, 15])
        block_minutes = rng.randint(10, 30)
        people.extend([count] * (block_minutes * 60 // interval_seconds))
    return people[:steps]


def noisy_count(mean, rng):
    """An integer count, never negative, around 'mean'."""
    return max(0, round(rng.gauss(mean, math.sqrt(mean) + 1)))


def simulate_readings(start_time, minutes, interval_seconds, rng):
    """Simulate the messages the node would send, one per time step.

    Returns two lists:
    - messages: each message has 'timestamp', the sensors, the label
      (busy_level, 0 to 3, set by a person) and 'pirs'.
    - labels: the true number of people and the level at each moment.
    """
    start = datetime.strptime(start_time, TIME_FORMAT)
    steps = minutes * 60 // interval_seconds
    people = simulate_people(steps, interval_seconds, rng)

    # Initial values of the slowly changing sensors
    co2, temperature, humidity = 450.0, 21.0, 45.0
    messages, labels = [], []

    for i in range(steps):
        timestamp = format_time(start + timedelta(seconds=i * interval_seconds))
        n = people[i]

        # Each sensor slowly approaches a level that depends on the people
        co2 += (450 + 40 * n - co2) * 0.02 + rng.gauss(0, 5)
        temperature += (21 + 0.1 * n - temperature) * 0.01 + rng.gauss(0, 0.02)
        humidity += (45 + 0.5 * n - humidity) * 0.01 + rng.gauss(0, 0.1)

        # Probability that a PIR detects movement in one reading
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
    """Add on purpose the faults that the cleaning must fix.

    Returns the faulty messages and a list describing each fault.
    """
    messages = copy.deepcopy(messages)
    faults = []

    # 1. Long gap: the node stops sending for 3 minutes
    gap_length = 3 * 60 // interval_seconds
    start = rng.randint(100, len(messages) - gap_length - 100)
    faults.append({"type": "long_gap", "from": messages[start]["timestamp"],
                   "to": messages[start + gap_length - 1]["timestamp"]})
    del messages[start:start + gap_length]

    # 2. Short gaps: single messages are lost (with QoS 0 they are not resent)
    for index in sorted(rng.sample(range(10, len(messages) - 10), 3), reverse=True):
        faults.append({"type": "short_gap", "timestamp": messages[index]["timestamp"]})
        del messages[index]

    # 3. Impossible values in a specific sensor
    impossible = [("co2Equivalent", -50.0), ("co2Equivalent", -50.0),
                  ("temperature", 150.0), ("humidity", 130.0)]
    for sensor, value in impossible:
        index = rng.randrange(len(messages))
        messages[index][sensor] = value
        faults.append({"type": "impossible_value", "sensor": sensor,
                       "timestamp": messages[index]["timestamp"]})

    # 4. Isolated spikes in the microphone
    for index in rng.sample(range(1, len(messages) - 1), 4):
        messages[index]["mic_vol"] = 1.0
        faults.append({"type": "spike", "sensor": "mic_vol",
                       "timestamp": messages[index]["timestamp"]})

    # 5. Duplicates: the same message arrives twice (unlikely with QoS 0,
    #    but possible if QoS 1 is ever used)
    for index in sorted(rng.sample(range(len(messages)), 8), reverse=True):
        messages.insert(index + 1, copy.deepcopy(messages[index]))
        faults.append({"type": "duplicate", "timestamp": messages[index]["timestamp"]})

    return messages, faults


def to_api_response(messages):
    """Convert the messages to the API response format (openapi.yaml)."""
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
    """Generate the simulated data and save it to data/simulated/."""
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

    print(f"{len(messages)} simulated messages, {len(faults)} faults added")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

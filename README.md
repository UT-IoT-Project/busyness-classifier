# Room occupancy level classification (ML)

*Versión en español: [README.es.md](README.es.md)*

ML part of the IoT project. It estimates the **occupancy level** of a room,
from 0 to 3, using the sensors of the ESP32 nodes:

| Level | Meaning |
| --- | --- |
| 0 | Unoccupied |
| 1 | Slightly busy |
| 2 | Busy |
| 3 | Extremely busy |

It is **supervised classification**: the model is trained on data in which a
person sets the true level (`busy_level`), and then predicts it without them.
It reads the data from the API of the `platform` repository (contract in
`platform/docs/api/openapi.yaml`). The code and the model run on a laptop;
predictions are saved locally in a CSV file.

## Pipeline

```
API (Raspberry Pi) -> cleaning -> integration -> windows -> Random Forest -> CSV
```

| Path | Content |
| --- | --- |
| `config.yaml` | All settings: API, sensors, cleaning, windows, prediction |
| `src/occupancy/api_client.py` | Request readings from the real API or the simulated data |
| `src/occupancy/download.py` | Download a real data session to `data/raw/` |
| `src/occupancy/cleaning.py` | Phase 1: duplicates, range, spikes, gaps and smoothing |
| `src/occupancy/integration.py` | Phase 2: join sensors, PIRs and label into one table |
| `src/occupancy/features.py` | Phase 4: 3-minute windows and their features |
| `src/occupancy/train.py` | Train, validate leaving one session out, save the model |
| `src/occupancy/predict.py` | Live prediction every 30 s |
| `src/occupancy/publish.py` | Save each prediction to the CSV |
| `src/occupancy/simulate.py` | Generate simulated data to test without the API |
| `notebooks/` | Step-by-step notebooks for each phase (01 cleaning ... 04 model, in Spanish) |
| `tests/` | Tests (`python -m pytest`) |
| `data/simulated/` | Simulated data (committed); never mix with real data |
| `data/raw/` | Downloaded real sessions (not committed) |
| `data/predictions/` | Predictions (not committed) |
| `models/` | Trained model (not committed) |

Phase 3 (reduction) is done by looking at the feature importances shown in
training. Phase 5 (discretization) is not needed: the levels are already
discrete.

## Setup (Windows)

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Before running the modules, tell Python where the code is (only lasts while
the terminal is open):

- PowerShell (VS Code's default terminal): `$env:PYTHONPATH = "src"`
- Command Prompt (cmd): `set PYTHONPATH=src`

Before using the API, fill in the `api` section of `config.yaml`: Raspberry Pi
address, building and room.

## Usage

1. **Collect data**: with the nodes sending and someone setting `busy_level`,
   download each session (Amsterdam time). Each session should go through all
   four levels:

   ```sh
   python -m occupancy.download --session session1 --start "2026-11-03 10:00" --end "2026-11-03 12:30"
   ```

2. **Train** with all sessions in `data/raw/` (the more, the better; with two or
   more, validation leaves one whole session out each time):

   ```sh
   python -m occupancy.train
   ```

3. **Predict live** (one prediction every 30 s, saved to `data/predictions/predictions.csv`):

   ```sh
   python -m occupancy.predict
   ```

To test everything without the API: `python -m occupancy.train --simulated` and
`python -m occupancy.predict --simulate`.

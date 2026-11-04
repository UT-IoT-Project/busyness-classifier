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

## How it works

The phases follow the data preprocessing process of Lecture 7 (slide 24). All
values below are in `config.yaml` and are provisional until there is real data
and the sensor datasheets.

### 1. Cleaning (`cleaning.py`)

Each sensor is cleaned on its own, in this order. The label `busy_level` only
goes through duplicates, range check and the time grid: it is never
interpolated or smoothed.

| Step | What it does | Setting |
| --- | --- | --- |
| 1.1 Duplicates | Removes rows that are exactly equal (a message delivered twice) | - |
| 1.2 Range check | Values outside the physical limits become empty | `limits`, `label` |
| 1.3 Isolated spikes | A value further than a threshold from the centred median of its neighbours becomes empty. Not applied to `wifiPackets` and `bleAdverts`, which are counts and can really jump | `spike_window_readings`, `spike_thresholds` |
| 1.4 Gaps | Puts every sensor on a common time grid; gaps of up to 3 readings are interpolated, longer gaps stay completely empty | `interval_seconds`, `max_gap_readings` |
| 1.5 Smoothing | Optional backward-looking moving average; does not fill gaps | `smoothing` |

Current values:

| Setting | Value |
| --- | --- |
| `interval_seconds` | 10 s |
| `limits` | temperature -10 to 50, humidity 0 to 100, co2Equivalent 0 to 5000, pressure 950 to 1100, voc >= 0, iaq 0 to 500, mic_vol -100 to 0 (dBFS), wifiPackets and bleAdverts 0 to 4294967295 |
| `label` | busy_level, 0 to 3 |
| `spike_window_readings` | 5 readings |
| `spike_thresholds` | temperature 1, humidity 2, pressure 3, co2Equivalent 100, voc 0.5, iaq 100, mic_vol 10 (dB) |
| `max_gap_readings` | 3 readings (30 s) |
| `smoothing` | enabled, 30 s, on mic_vol, co2Equivalent and voc |

Slides (Lecture 7): 21-23 data quality, 25 smoothing, 26 incomplete data.

With the simulated data (180 min, one reading every 10 s, with 8 duplicates,
4 impossible values, 4 mic_vol spikes, 3 short gaps and one 3-minute gap) the
clean table has 1080 rows, and in each sensor only the 18 readings of the long
gap stay empty.

### 2. Integration (`integration.py`)

`merge_all` joins the sensor tables, the PIR table (one column `pir-<ID>` per
PIR) and the label into one table with `pd.concat(axis=1)`, indexed by
timestamp. This works because cleaning already put everything on the same
grid. With the simulated data: 1080 rows x 13 columns. Correlation between
sensors will be checked with real data (slides 27-33).

### 3. Reduction

Nothing is removed by hand. At the end of training, `train.py` prints the
importance of each feature (`feature_importances_`, they add up to 1); those close to zero are candidates to remove, checking that
accuracy does not drop (slides 34-46). The decision is made with real data.

### 4. Transformation: windows and features (`features.py`)

One **3-minute** window every **30 seconds**, looking only backwards, so it is
computed the same way in training and live (slides 46-47). No normalization
is needed: Random Forest is not affected by scale (slide 48).

| Feature | Meaning |
| --- | --- |
| `<sensor>_mean` | Mean in the window of pressure, humidity, temperature, co2Equivalent, voc, iaq, mic_vol, wifiPackets and bleAdverts |
| `co2Equivalent_change` | Change in CO2 compared with 3 minutes before |
| `mic_vol_max` | Maximum noise |
| `mic_vol_std` | How much the noise varies |
| `pir_mean` | Mean activity of all PIRs |
| `pir_max` | Maximum activity of any PIR |
| `label` | Most frequent busy_level in the window (training only) |

A window is valid if it is complete, if at least half of the sensors
(`min_sensor_fraction`) have data in 80 % of the window (`min_coverage`) and,
in training, if it has a label. A sensor without data leaves its features
empty and the window is still used. PIRs are found by name, however many there
are. The simulated data gives 345 windows (level 0: 201, 1: 22, 2: 45, 3: 77).

### 5. Discretization

Not needed: the levels already arrive discrete (slide 49).

### Model and validation (`train.py`)

`RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`
(slides 65-69). Empty feature values are filled with the training median.

Validation leaves one whole collection session out each time
(`LeaveOneGroupOut`). With a single session, such as the simulated data,
15-minute blocks are used instead (`StratifiedGroupKFold`), which is less
reliable. It reports accuracy, accuracy within one level, the confusion matrix
and precision/recall/F1 per level, and warns if a level is missing or a session
does not cover all four levels. Simulated data: accuracy 0.98 (made-up, easy
data; it says nothing about real data).

`models/model.joblib` contains:

| Key | Use |
| --- | --- |
| `model` | Random Forest trained on all data |
| `features` | Names and order of the feature columns |
| `fill_values` | Median of each feature, to fill empty values live |
| `feature_min`, `feature_max` | Range seen in training, to flag values never seen |

### Live prediction (`predict.py`)

Every 30 s:

1. **Get the data**: asks the API for the window plus 3 minutes of history
   (`extra_history_minutes`) and a 10-minute margin (`clock_margin_minutes`) in
   case the laptop and Raspberry Pi clocks differ. A 404 means no data.
2. **Run the data through the phases**, with the same `config.yaml` as in
   training.
3. **Use the model** on the last window, which ends at the last reading
   received (not at the laptop's time). If nothing new arrived since the
   previous prediction, it skips.
4. **Build the result** and append it to the CSV.

A prediction is marked as unreliable (`reliable` = False) if the probability of
the chosen level is below 0.6 (`min_confidence`), if a sensor is missing or if
a feature is outside the range seen in training. If a request fails, the error
is shown and the loop continues.

### Output (`publish.py`)

Each prediction adds a row to `data/predictions/predictions.csv` with
`timestamp`, `predicted_level`, `confidence`, `prob_0` to `prob_3`, `reliable`
and `real_level` (the label, if it arrives). With `--simulate` it writes to
`predictions_simulated.csv`.

## Setup (Windows)

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Before using the API, fill in the `api` section of `config.yaml`: Raspberry Pi
address, building and room.

## Usage

1. **Collect data**: with the nodes sending and someone setting `busy_level`,
   download each session (Amsterdam time). Each session should go through all
   four levels:

   ```sh
   python src\occupancy\download.py --session session1 --start "2026-11-03 10:00" --end "2026-11-03 12:30"
   ```

2. **Train** with all sessions in `data/raw/` (the more, the better; with two or
   more, validation leaves one whole session out each time):

   ```sh
   python src\occupancy\train.py
   ```

3. **Predict live** (one prediction every 30 s, saved to `data/predictions/predictions.csv`):

   ```sh
   python src\occupancy\predict.py
   ```

   At each step it shows 1) the request sent to the API, 2) what the API answers
   (HTTP code and, per sensor, number of readings, first and last time and last
   value) and 3) what goes into the model (each feature next to the range seen in
   training, marking the ones outside it), the probability of each level and the
   result, with the reason if it is unreliable. With `--quiet` it shows only one
   line per prediction.

To test everything without the API: `python src\occupancy\train.py --simulated` and
`python src\occupancy\predict.py --simulate`.

## Open decisions

- [ ] The firmware still sends `busynessLabel`; it must send `busy_level`. Until then there is no real labelled data.
- [ ] Plan the collection sessions: at least two, each covering all four levels.
- [ ] Real Raspberry Pi address, building and room in `config.yaml` (`api`).
- [ ] Sensor units and datasheets, to set the range limits.
- [ ] With plots of real data: grid interval, spike thresholds and smoothing window; compare the model with and without smoothing.
- [ ] Window size (between 1 and 5 min; now 3).
- [ ] Which features to remove based on their importance with real data.
- [ ] Confirm with the professor the Thingy requirement (IMU, Bluetooth, on-device ML); the group decided not to use them.
- [ ] Messages do not carry a node identifier.

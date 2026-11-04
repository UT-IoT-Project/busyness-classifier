# Clasificacion del nivel de ocupacion de salas (ML)

*English version: [README.md](README.md)*

Parte de ML del proyecto de IoT. Estima el **nivel de ocupacion** de una sala,
de 0 a 3, a partir de los sensores de los nodos ESP32:

| Nivel | Significado |
| --- | --- |
| 0 | Desocupada |
| 1 | Un poco ocupada |
| 2 | Ocupada |
| 3 | Extremadamente ocupada |

Es una **clasificacion supervisada**: se entrena con datos en los que una
persona marca el nivel real (`busy_level`) y despues lo predice sin ella.
Lee los datos de la API del repositorio `platform` (contrato en
`platform/docs/api/openapi.yaml`). El codigo y el modelo se ejecutan en un
portatil; las predicciones se guardan en local, en un CSV.

## Pipeline

```
API (Raspberry Pi) -> limpieza -> integracion -> ventanas -> Random Forest -> CSV
```

| Ruta | Contenido |
| --- | --- |
| `config.yaml` | Toda la configuracion: API, sensores, limpieza, ventanas, prediccion |
| `src/occupancy/api_client.py` | Pedir lecturas a la API real o a los datos simulados |
| `src/occupancy/download.py` | Descargar una sesion de datos reales a `data/raw/` |
| `src/occupancy/cleaning.py` | Fase 1: duplicados, rango, picos, huecos y suavizado |
| `src/occupancy/integration.py` | Fase 2: unir sensores, PIR y etiqueta en una tabla |
| `src/occupancy/features.py` | Fase 4: ventanas de 3 min y sus caracteristicas |
| `src/occupancy/train.py` | Entrenar, validar dejando fuera una sesion y guardar el modelo |
| `src/occupancy/predict.py` | Prediccion en vivo cada 30 s |
| `src/occupancy/publish.py` | Guardar cada prediccion en el CSV |
| `src/occupancy/simulate.py` | Generar datos simulados para probar sin la API |
| `notebooks/` | Notebooks paso a paso de cada fase (01 limpieza ... 04 modelo) |
| `tests/` | Pruebas (`python -m pytest`) |
| `data/simulated/` | Datos simulados (se suben); nunca mezclar con los reales |
| `data/raw/` | Sesiones reales descargadas (no se suben) |
| `data/predictions/` | Predicciones (no se suben) |
| `models/` | Modelo entrenado (no se sube) |

## Como funciona

Las fases siguen el proceso de preprocesado de datos de la clase 7
(diapositiva 24). Todos los valores estan en `config.yaml` y son provisionales
hasta tener datos reales y las hojas de datos de los sensores.

### 1. Limpieza (`cleaning.py`)

Cada sensor se limpia por separado, en este orden. La etiqueta `busy_level`
solo pasa por duplicados, control de rango y la rejilla de tiempo: nunca se
interpola ni se suaviza.

| Paso | Que hace | Parametro |
| --- | --- | --- |
| 1.1 Duplicados | Quita las filas exactamente iguales (un mensaje que llega dos veces) | - |
| 1.2 Control de rango | Los valores fuera de los limites fisicos se quedan vacios | `limits`, `label` |
| 1.3 Picos aislados | Un valor que se aleja mas de un umbral de la mediana centrada de sus vecinos se queda vacio. No se aplica a `wifiPackets` ni `bleAdverts`, que son conteos y pueden saltar de verdad | `spike_window_readings`, `spike_thresholds` |
| 1.4 Huecos | Pone todos los sensores en una rejilla de tiempo comun; los huecos de hasta 3 lecturas se interpolan y los mas largos se dejan enteros vacios | `interval_seconds`, `max_gap_readings` |
| 1.5 Suavizado | Media movil opcional que solo mira hacia atras; no rellena huecos | `smoothing` |

Valores actuales:

| Parametro | Valor |
| --- | --- |
| `interval_seconds` | 10 s |
| `limits` | temperature -10 a 50, humidity 0 a 100, co2Equivalent 0 a 5000, pressure 950 a 1100, voc >= 0, iaq 0 a 500, mic_vol -100 a 0 (dBFS), wifiPackets y bleAdverts 0 a 4294967295 |
| `label` | busy_level, de 0 a 3 |
| `spike_window_readings` | 5 lecturas |
| `spike_thresholds` | temperature 1, humidity 2, pressure 3, co2Equivalent 100, voc 0,5, iaq 100, mic_vol 10 (dB) |
| `max_gap_readings` | 3 lecturas (30 s) |
| `smoothing` | activado, 30 s, sobre mic_vol, co2Equivalent y voc |

Diapositivas (clase 7): 21-23 calidad de los datos, 25 suavizado, 26 datos incompletos.

Con los datos simulados (180 min, una lectura cada 10 s, con 8 duplicados,
4 valores imposibles, 4 picos de mic_vol, 3 huecos cortos y un hueco de
3 minutos) la tabla limpia tiene 1080 filas y en cada sensor solo quedan vacias
las 18 lecturas del hueco largo.

### 2. Integracion (`integration.py`)

`merge_all` une con `pd.concat(axis=1)` las tablas de los sensores, la de los
PIR (una columna `pir-<ID>` por PIR) y la etiqueta en una sola tabla, con el
timestamp como indice. Funciona porque la limpieza ya puso todo en la misma
rejilla. Con los datos simulados: 1080 filas x 13 columnas. La correlacion
entre sensores se revisara con datos reales (diapositivas 27-33).

### 3. Reduccion

No se quita nada a mano. Al final del entrenamiento, `train.py` imprime la
importancia de cada caracteristica (`feature_importances_`, suman 1); las que salen casi a cero son candidatas a quitarse,
comprobando que la exactitud no baja (diapositivas 34-46). La decision se toma
con datos reales.

### 4. Transformacion: ventanas y caracteristicas (`features.py`)

Una ventana de **3 minutos** cada **30 segundos**, que solo mira hacia atras,
para que se calcule igual al entrenar y en directo (diapositivas 46-47). No
hace falta normalizar: al Random Forest no le afecta la escala (diapositiva 48).

| Caracteristica | Que es |
| --- | --- |
| `<sensor>_mean` | Media en la ventana de pressure, humidity, temperature, co2Equivalent, voc, iaq, mic_vol, wifiPackets y bleAdverts |
| `co2Equivalent_change` | Cuanto ha cambiado el CO2 respecto a 3 minutos antes |
| `mic_vol_max` | Ruido maximo |
| `mic_vol_std` | Cuanto varia el ruido |
| `pir_mean` | Actividad media de todos los PIR |
| `pir_max` | Actividad maxima de cualquier PIR |
| `label` | busy_level mas frecuente en la ventana (solo al entrenar) |

Una ventana es valida si esta completa, si al menos la mitad de los sensores
(`min_sensor_fraction`) tienen datos en el 80 % de la ventana (`min_coverage`)
y, al entrenar, si tiene etiqueta. Si un sensor no tiene datos, sus
caracteristicas quedan vacias y la ventana se usa igual. Los PIR se buscan por
nombre, haya los que haya. Los datos simulados dan 345 ventanas (nivel 0: 201,
1: 22, 2: 45, 3: 77).

### 5. Discretizacion

No hace falta: los niveles ya llegan discretizados (diapositiva 49).

### Modelo y validacion (`train.py`)

`RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`
(diapositivas 65-69). Los valores vacios de las caracteristicas se rellenan con
la mediana del entrenamiento.

La validacion deja fuera una sesion de recogida entera cada vez
(`LeaveOneGroupOut`). Con una sola sesion, como los datos simulados, se usan
bloques de 15 minutos (`StratifiedGroupKFold`), que es menos fiable. Muestra la
exactitud, el acierto con margen de un nivel, la matriz de confusion y
precision/recall/F1 por nivel, y avisa si falta algun nivel o si una sesion no
cubre los cuatro. Datos simulados: exactitud 0,98 (datos inventados y faciles;
no dicen nada de los reales).

`models/model.joblib` contiene:

| Clave | Para que |
| --- | --- |
| `model` | El Random Forest entrenado con todos los datos |
| `features` | Nombres y orden de las columnas |
| `fill_values` | Mediana de cada caracteristica, para rellenar vacios en directo |
| `feature_min`, `feature_max` | Rango visto al entrenar, para avisar de valores nunca vistos |

### Prediccion en directo (`predict.py`)

Cada 30 s:

1. **Obtener los datos**: pide a la API la ventana mas 3 minutos de historia
   (`extra_history_minutes`) y 10 minutos de margen (`clock_margin_minutes`)
   por si los relojes del portatil y de la Raspberry Pi no coinciden. Un 404
   significa que no hay datos.
2. **Pasar los datos por las fases**, con la misma `config.yaml` que al
   entrenar.
3. **Usar el modelo** sobre la ultima ventana, que termina en la ultima
   lectura recibida (no en la hora del portatil). Si no ha llegado nada nuevo
   desde la prediccion anterior, no predice.
4. **Construir el resultado** y anadirlo al CSV.

Una prediccion se marca como no fiable (`reliable` = False) si la probabilidad
del nivel elegido es menor de 0,6 (`min_confidence`), si falta algun sensor o
si alguna caracteristica esta fuera del rango visto al entrenar. Si una llamada
falla, se muestra el error y el bucle sigue.

### Salida (`publish.py`)

Cada prediccion anade una fila a `data/predictions/predictions.csv` con
`timestamp`, `predicted_level`, `confidence`, `prob_0` a `prob_3`, `reliable`
y `real_level` (la etiqueta, si llega). Con `--simulate` se escribe en
`predictions_simulated.csv`.

## Puesta en marcha (Windows)

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Antes de ejecutar los modulos, indica a Python donde esta el codigo (solo dura
mientras la terminal este abierta):

- PowerShell (la terminal por defecto de VS Code): `$env:PYTHONPATH = "src"`
- Simbolo del sistema (cmd): `set PYTHONPATH=src`

Antes de usar la API, completa en `config.yaml` la seccion `api`: direccion de
la Raspberry Pi, edificio y sala.

## Uso

1. **Recoger datos**: con los nodos enviando y una persona marcando `busy_level`,
   descarga cada sesion (horas de Amsterdam):

   ```sh
   python -m occupancy.download --session session1 --start "2026-11-03 10:00" --end "2026-11-03 12:30"
   ```

2. **Entrenar** con todas las sesiones de `data/raw/` (cuantas mas, mejor; con
   dos o mas, la validacion deja fuera una sesion entera cada vez):

   ```sh
   python -m occupancy.train
   ```

3. **Predecir en vivo** (una prediccion cada 30 s, en `data/predictions/predictions.csv`):

   ```sh
   python -m occupancy.predict
   ```

Para probar todo sin la API: `python -m occupancy.train --simulated` y
`python -m occupancy.predict --simulate`.

## Decisiones abiertas

- [ ] El firmware todavia envia `busynessLabel`; tiene que enviar `busy_level`. Hasta entonces no hay datos reales etiquetados.
- [ ] Planificar las sesiones de recogida: al menos dos, y cada una con los cuatro niveles.
- [ ] Direccion real de la Raspberry Pi, edificio y sala en `config.yaml` (`api`).
- [ ] Unidades y hojas de datos de los sensores, para fijar los limites de rango.
- [ ] Con graficas de datos reales: intervalo de la rejilla, umbrales de picos y ventana del suavizado; comparar el modelo con y sin suavizado.
- [ ] Tamano de la ventana (entre 1 y 5 min; ahora 3).
- [ ] Que caracteristicas quitar segun su importancia con datos reales.
- [ ] Confirmar con el profesor el requisito de los Thingy (IMU, Bluetooth, ML en el dispositivo); el grupo decidio no usarlos.
- [ ] Los mensajes no traen identificador del nodo.

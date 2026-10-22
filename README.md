# Estimacion de ocupacion de salas (ML)

Modelo que estima el numero de personas en una sala a partir de los sensores
de los nodos ESP32. Lee los datos de la API del repositorio `platform`.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `config.yaml` | Configuracion: API, sala, sensores, ventanas, limpieza |
| `src/occupancy/api_client.py` | Pedir lecturas a la API |
| `src/occupancy/simulate.py` | Generar datos simulados para probar sin la API |
| `src/occupancy/cleaning.py` | Fase 1: limpieza (duplicados, rangos, huecos, suavizado) |
| `src/occupancy/integration.py` | Fase 2: unir todos los sensores en una tabla |
| `src/occupancy/reduction.py` | Fase 3: elegir caracteristicas con datos reales |
| `src/occupancy/features.py` | Fase 4: ventanas de tiempo y caracteristicas |
| `src/occupancy/labels.py` | Etiquetas: numero real de personas |
| `src/occupancy/train.py` | Entrenamiento |
| `src/occupancy/evaluate.py` | Validacion y metricas |
| `src/occupancy/predict.py` | Prediccion en vivo |
| `src/occupancy/publish.py` | Guardar la prediccion |
| `data/` | Datos locales: `raw`, `processed` y `labels` no se suben |
| `data/simulated/` | Datos simulados (si se suben); nunca mezclar con los reales |
| `models/` | Modelos entrenados (no se suben) |
| `notebooks/` | Exploracion de datos |
| `tests/` | Pruebas |

La fase 5 (discretizacion) no se usa: la salida es un numero, no una categoria.

## Puesta en marcha

```sh
python -m venv .venv
.venv\Scripts\activate        # en Windows
pip install -r requirements.txt
```

Para ejecutar los modulos desde la raiz: `set PYTHONPATH=src` (Windows) y
despues `python -m occupancy.train` o `python -m occupancy.predict`.

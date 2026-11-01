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

La fase 3 (reduccion) se hace mirando la importancia de cada caracteristica que
muestra el entrenamiento. La fase 5 (discretizacion) no hace falta: los niveles
ya llegan discretizados.

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

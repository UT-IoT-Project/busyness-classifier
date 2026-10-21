"""Inferencia en vivo. Se ejecuta con: python -m occupancy.predict"""


def predict_now(config, model):
    """Pide la ultima ventana a la API, aplica el mismo preprocesado y predice."""
    raise NotImplementedError


def main():
    """Bucle: cada cierto tiempo, predecir y guardar el resultado."""
    raise NotImplementedError


if __name__ == "__main__":
    main()

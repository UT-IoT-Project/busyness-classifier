"""Carga la configuracion de config.yaml."""
import yaml


def load_config(path="config.yaml"):
    """Lee config.yaml y lo devuelve como diccionario."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

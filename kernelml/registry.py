"""
Almacenamiento de modelos entrenados (punto 5a).

Un modelo se guarda como un unico `<id>.joblib` con {"pipeline", "discretizer", "meta"} y un
`<id>.json` con los mismos metadatos (para listar sin cargar el modelo).  Es el formato que lee la
aplicacion web y el que el notebook usa para exportar los mejores pipelines.
"""
import json
import os
import re

import joblib

REQUIRED_META = ("id", "name", "dataset", "metrics", "confusion_matrix", "class_names", "class_ranges")


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_model_bundle(pipeline, discretizer, meta, out_dir):
    """Guarda pipeline + discretizador + metadatos. `meta` debe traer al menos REQUIRED_META y `features`."""
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"Faltan metadatos: {missing}")
    meta = dict(meta, id=slugify(meta["id"]))
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "discretizer": discretizer, "meta": meta},
                os.path.join(out_dir, meta["id"] + ".joblib"))
    with open(os.path.join(out_dir, meta["id"] + ".json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=1)
    return meta["id"]


def load_model_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"pipeline", "discretizer", "meta"} <= set(bundle):
        raise ValueError("El archivo no es un bundle valido (se esperaba {pipeline, discretizer, meta})")
    missing = [k for k in REQUIRED_META if k not in bundle["meta"]]
    if missing:
        raise ValueError(f"Faltan metadatos: {missing}")
    return bundle

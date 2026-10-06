"""
Aplicacion web: clasificacion de zonas de irradiancia con modelos kernel.

  a) Almacena modelos (carpeta webapp/models; se precarga con webapp/seed_models; permite subir/borrar).
  b) Compara dos modelos: mapas lado a lado + metricas + matrices de confusion.
  c) Al picar sobre el mapa muestra la clase de irradiancia de ese punto segun los modelos comparados.

Ejecutar:  python webapp/app.py   ->  http://127.0.0.1:5000
"""
import json
import os
import shutil
import sys
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, abort
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # para importar el paquete kernelml (clases de los modelos)

from kernelml.registry import load_model_bundle, slugify   # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
SEED_DIR = os.path.join(HERE, "seed_models")
DATA_DIR = os.path.join(HERE, "data")
DATASET_FILES = {"landsat": "landsat_model.csv", "modis": "modis_model.csv"}
BANDS = [f"band{i}" for i in range(1, 8)]
MAX_UPLOAD_MB = 50
NEAREST_MAX_KM = 10      # si el punto no cae en una celda con datos, se usa la celda mas cercana hasta esta distancia

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
_to_m = Transformer.from_crs(4326, 3857, always_xy=True)
_lock = threading.Lock()
_bundles = {}      # id -> bundle cargado
_maps = {}         # id -> mapa precalculado
_datasets = {}     # dataset -> dict con DataFrame y geometria de celdas


# ---------------------------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------------------------
def get_dataset(name):
    """Carga un CSV. Las columnas 'latitude'/'longitude' son en realidad X/Y en EPSG:3857 (metros)."""
    if name in _datasets:
        return _datasets[name]
    if name not in DATASET_FILES:
        abort(404, f"Dataset desconocido: {name}")
    df = pd.read_csv(os.path.join(DATA_DIR, DATASET_FILES[name]))
    x, y = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    step_x = np.min(np.diff(np.unique(x)))
    step_y = np.min(np.diff(np.unique(y)))
    half_x, half_y = step_x / 2, step_y / 2
    lon, lat = _to_ll.transform(x, y)
    west, south = _to_ll.transform(x - half_x, y - half_y)
    east, north = _to_ll.transform(x + half_x, y + half_y)
    ds = dict(df=df, x=x, y=y, half_x=half_x, half_y=half_y, lon=lon, lat=lat,
              bounds=np.column_stack([south, west, north, east]))
    _datasets[name] = ds
    return ds


# ---------------------------------------------------------------------------------------------
# Almacen de modelos
# ---------------------------------------------------------------------------------------------
def init_store():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.isdir(SEED_DIR):
        for f in os.listdir(SEED_DIR):                          # copia los modelos semilla que falten
            if not os.path.exists(os.path.join(MODELS_DIR, f)):
                shutil.copy(os.path.join(SEED_DIR, f), os.path.join(MODELS_DIR, f))


def list_models():
    out = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(MODELS_DIR, f), encoding="utf8") as fh:
                out.append(json.load(fh))
    return out


def get_bundle(model_id):
    path = os.path.join(MODELS_DIR, slugify(model_id) + ".joblib")
    if not os.path.exists(path):
        abort(404, f"Modelo no encontrado: {model_id}")
    with _lock:
        if model_id not in _bundles:
            _bundles[model_id] = load_model_bundle(path)
        return _bundles[model_id]


def predict_cells(model_id):
    """Clase predicha por el modelo para todas las celdas de su dataset (se calcula una vez)."""
    if model_id in _maps:
        return _maps[model_id]
    b = get_bundle(model_id)
    meta = b["meta"]
    ds = get_dataset(meta["dataset"])
    feats = meta.get("features", BANDS)
    pred = b["pipeline"].predict(ds["df"][feats].to_numpy())
    real = b["discretizer"].transform(ds["df"]["value"].to_numpy())
    res = dict(pred=[int(v) for v in pred], real=[int(v) for v in real])
    _maps[model_id] = res
    return res


def class_info(meta, k):
    r = meta["class_ranges"][k] if k < len(meta["class_ranges"]) else None
    return {"index": int(k), "name": meta["class_names"][k] if k < len(meta["class_names"]) else str(k),
            "range": r}


# ---------------------------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    return jsonify(list_models())


@app.post("/api/models")
def api_upload():
    """Sube un bundle .joblib creado con kernelml.registry.save_model_bundle (ver README)."""
    f = request.files.get("file")
    if f is None or not f.filename.endswith(".joblib"):
        return jsonify(error="Sube un archivo .joblib"), 400
    tmp = os.path.join(MODELS_DIR, "_upload.tmp.joblib")
    f.save(tmp)
    try:
        bundle = load_model_bundle(tmp)
        meta = bundle["meta"]
        if meta["dataset"] not in DATASET_FILES:
            raise ValueError(f"dataset desconocido: {meta['dataset']}")
        mid = slugify(meta["id"])
        meta["id"] = mid
        os.replace(tmp, os.path.join(MODELS_DIR, mid + ".joblib"))
        with open(os.path.join(MODELS_DIR, mid + ".json"), "w", encoding="utf8") as fh:
            json.dump(meta, fh, indent=1)
        _bundles.pop(mid, None)
        _maps.pop(mid, None)
        return jsonify(meta), 201
    except Exception as e:                                         # archivo invalido
        if os.path.exists(tmp):
            os.remove(tmp)
        return jsonify(error=f"Modelo invalido: {e}"), 400


@app.delete("/api/models/<model_id>")
def api_delete(model_id):
    mid = slugify(model_id)
    found = False
    for ext in (".joblib", ".json"):
        p = os.path.join(MODELS_DIR, mid + ext)
        if os.path.exists(p):
            os.remove(p)
            found = True
    _bundles.pop(mid, None)
    _maps.pop(mid, None)
    if not found:
        abort(404)
    return jsonify(deleted=mid)


@app.get("/api/models/<model_id>/download")
def api_download(model_id):
    p = os.path.join(MODELS_DIR, slugify(model_id) + ".joblib")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, as_attachment=True)


@app.get("/api/models/<model_id>/map")
def api_map(model_id):
    """Celdas del dataset del modelo con su clase predicha y real."""
    meta = get_bundle(model_id)["meta"]
    ds = get_dataset(meta["dataset"])
    m = predict_cells(model_id)
    return jsonify(
        meta=meta,
        bounds=np.round(ds["bounds"], 6).tolist(),
        pred=m["pred"], real=m["real"],
        value=ds["df"]["value"].round(2).tolist(),
        center=[float(np.mean(ds["lat"])), float(np.mean(ds["lon"]))],
    )


@app.get("/api/predict")
def api_predict():
    """Clasifica el punto (lat, lon) con cada modelo pedido en ?ids=a,b."""
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return jsonify(error="Parametros lat y lon requeridos"), 400
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    px, py = _to_m.transform(lon, lat)
    out = {}
    for mid in ids:
        b = get_bundle(mid)
        meta = b["meta"]
        ds = get_dataset(meta["dataset"])
        dist = np.hypot(ds["x"] - px, ds["y"] - py)
        inside = (np.abs(ds["x"] - px) <= ds["half_x"]) & (np.abs(ds["y"] - py) <= ds["half_y"])
        exact = bool(inside.any())
        if not exact and dist.min() / 1000 > NEAREST_MAX_KM:        # fuera de la zona de estudio
            out[mid] = dict(found=False, name=meta["name"], dataset=meta["dataset"],
                            nearest_km=round(float(dist.min()) / 1000, 1))
            continue
        i = int(np.argmin(np.where(inside, dist, np.inf))) if exact else int(np.argmin(dist))
        row = ds["df"].iloc[[i]]
        feats = meta.get("features", BANDS)
        k = int(b["pipeline"].predict(row[feats].to_numpy())[0])
        real = int(b["discretizer"].transform(row["value"].to_numpy())[0])
        out[mid] = dict(found=True, name=meta["name"], dataset=meta["dataset"], cell=i,
                        predicted=class_info(meta, k), real=class_info(meta, real),
                        value=float(row["value"].iloc[0]), correct=bool(k == real), exact=exact,
                        distance_km=round(float(dist[i]) / 1000, 1),
                        bands={c: float(row[c].iloc[0]) for c in feats})
    return jsonify(lat=lat, lon=lon, results=out)


init_store()

if __name__ == "__main__":
    # debug desactivado: el servidor de depuracion permite ejecutar codigo y los modelos son pickles
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)

# Taller 2 – Funciones kernel en la clasificación de zonas de irradiancia (Nariño)

Comparación de funciones kernel (Belanche) en la clasificación de irradiancia a partir de imágenes **Landsat** y **MODIS**,
con pipelines *Escalador → Reductor → Modelo(kernel)* sobre una variable objetivo discretizada, y una **aplicación web** que
almacena los mejores modelos, los compara (mapas + métricas) y clasifica cualquier punto que se pique en el mapa.

> Vídeo de demostración (1–2 min): _<pega aquí el enlace de YouTube>_

## Contenido

| Ruta | Descripción |
|---|---|
| `Taller2_Kernels.ipynb` | **Puntos 1–4**: datos, kernels, `KSVC`/`KANN`/`KRidgeClassifier`, 1008 configuraciones de pipeline, métricas, matrices de confusión, curvas PR, conclusiones y exportación de modelos |
| `kernelml/` | Paquete con kernels vectorizados (`KernelUtilities.py`), `KSVC` (`KSVM.py`), `KANNC` (`KANN.py`), **`KRidgeClassifier`** (`KRidge.py`), discretizadores, métricas y registro de modelos |
| `webapp/` | **Punto 5**: aplicación Flask + Leaflet (`app.py`, `templates/`, `static/`), CSV en `data/` y los 9 mejores modelos en `seed_models/` |
| `landsat_model.csv`, `modis_model.csv` | Datasets de entrada |
| `results/results_all.csv` | Resultado de las 1008 configuraciones (lo usa el notebook como caché) |

## Despliegue de la aplicación web

Requisitos: **Python 3.10+** (probado con 3.12) y conexión a internet en el navegador (el mapa base de OpenStreetMap y la
librería Leaflet se cargan desde CDN).

```bash
git clone <URL-de-este-repositorio>
cd <carpeta-del-repositorio>

python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

python webapp/app.py            # http://127.0.0.1:5000
```

> `requirements.txt` fija `scikit-learn==1.9.0`: los modelos `.joblib` son *pickles* y solo se cargan de forma fiable con la
> misma versión de scikit-learn con la que se guardaron. Si usas otra versión, vuelve a ejecutar el notebook (§8) para
> regenerar `webapp/seed_models/`.

Variables opcionales: `HOST` (por defecto `127.0.0.1`) y `PORT` (por defecto `5000`).

**Modo producción** (servidor WSGI en lugar del de desarrollo de Flask):

```bash
cd webapp
waitress-serve --listen=0.0.0.0:8000 app:app        # Windows / Linux / macOS
# Linux:  gunicorn -w 1 -b 0.0.0.0:8000 app:app     (pip install gunicorn)
```

Usa un solo *worker* (`-w 1`): los modelos y los mapas se cachean en memoria y las altas/bajas de modelos escriben en disco.

> ⚠️ **Seguridad.** Un `.joblib` es un pickle: **cargar un archivo malicioso ejecuta código**. La app no tiene autenticación;
> úsala en local o en una red de confianza y no la expongas a internet con la subida de modelos habilitada.

## Funcionalidades (punto 5)

1. **Almacenar los modelos** – pestaña *Modelos almacenados*. Al iniciar, la app copia los modelos de `webapp/seed_models/`
   a `webapp/models/` (el almacén real). Desde la web se pueden **subir** nuevos modelos `.joblib`, **descargarlos** y
   **eliminarlos**.
2. **Comparar dos modelos** – pestaña *Comparar modelos*: se eligen el modelo A y el B. Se muestran **dos mapas sincronizados**
   (zoom y desplazamiento) con la capa *clase predicha*, *clase real* o *aciertos/errores*, la **tabla de métricas**
   (Accuracy, F1-macro, AUC, MCC y F1 de validación cruzada, con el mejor resaltado), las **matrices de confusión** y los
   hiperparámetros de cada pipeline.
3. **Picar en el mapa** – al hacer clic en cualquier punto de cualquiera de los dos mapas, se muestra la **clase de irradiancia**
   (baja/media/alta, con su rango) que asigna cada modelo y la clase real del píxel. Detalle: los modelos usan las 7 bandas
   como *features*, así que se toman las bandas del píxel del dataset que contiene el punto; si el punto cae en un hueco
   entre píxeles se usa el **píxel más cercano (hasta 10 km)** y la app avisa de que es una estimación; más allá de 10 km se
   indica que está fuera de la zona de estudio.

### Guardar en la app un modelo propio

```python
from kernelml.registry import save_model_bundle
meta = dict(id="mi-modelo", name="Mi modelo", dataset="landsat",          # 'landsat' o 'modis'
            features=[f"band{i}" for i in range(1, 8)],
            metrics={"accuracy": .0, "f1_macro": .0, "auc_ovr": .0, "mcc": .0},
            confusion_matrix=[[...]], class_names=["baja", "media", "alta"],
            class_ranges=disc.ranges_)                                    # rangos de irradiancia por clase
save_model_bundle(pipeline, disc, meta, "webapp/seed_models")             # o súbelo desde la web
```

`pipeline` es un `sklearn.pipeline.Pipeline` ya entrenado sobre las 7 bandas y `disc` el `TargetDiscretizer` ajustado.

## Reproducir el notebook

```bash
jupyter notebook Taller2_Kernels.ipynb        # ejecutar desde la raíz del repositorio
```

Si existe `results/results_all.csv` el notebook lo reutiliza (segundos). Para recalcular las 1008 configuraciones borra ese
archivo: tarda ≈ 5 min en una máquina de 16 núcleos.

## Notas técnicas

* Las columnas `latitude`/`longitude` de los CSV son en realidad **X/Y en EPSG:3857 (metros)** y están intercambiadas; la app
  las convierte a WGS84 con `pyproj`.
* Los kernels se **vectorizaron** y se hicieron **serializables** respecto al código original de `KernelUtilities`
  (el notebook verifica que reproducen las funciones originales). En `KANN` se corrigió que `hyperbolic`/`rquadratic`
  ignoraran `gamma`/`coef0`. Ver §2 del notebook.
* Ni el shapefile de Nariño ni `gridNarinio.csv` (usados en `Regressor.ipynb`) estaban disponibles, por lo que el mapa se
  construye con las celdas de los propios datasets (no hay interpolación/kriging continuo ni recorte por el límite
  departamental).

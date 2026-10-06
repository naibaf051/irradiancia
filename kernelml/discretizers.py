"""
Discretizacion de la variable objetivo (irradiancia continua -> clases ordenadas 0=baja ... K-1=alta).

Metodos: 'uniform' (rangos de igual ancho), 'kmeans', 'hierarchical' (aglomerativo, Ward) y 'dbscan'.
El discretizador se ajusta SOLO con el y de entrenamiento y despues asigna clase a cualquier valor
nuevo (test, puntos del mapa) => no hay fuga de informacion hacia el conjunto de prueba.
Los metodos basados en clustering asignan los valores nuevos al cluster del vecino mas cercano
(1-NN sobre el y de entrenamiento), que en 1-D equivale a intervalos contiguos.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier

METHODS = ("uniform", "kmeans", "hierarchical", "dbscan")


class TargetDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, method="uniform", n_classes=3, random_state=0, dbscan_min_frac=0.05):
        self.method = method
        self.n_classes = n_classes
        self.random_state = random_state
        self.dbscan_min_frac = dbscan_min_frac     # min_samples de DBSCAN = fraccion de n

    # -- ajuste ---------------------------------------------------------------------------------
    def _dbscan_labels(self, y):
        n = len(y)
        ms = max(3, int(round(self.dbscan_min_frac * n)))
        span = y.max() - y.min()
        best = None
        for eps in np.linspace(span / 400, span / 4, 120):
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(y)
            k = len(set(lab)) - (1 if -1 in lab else 0)
            if k < 2:
                continue
            key = (abs(k - self.n_classes), int((lab == -1).sum()), eps)   # k cercano al pedido, poco ruido
            if best is None or key < best[0]:
                best = (key, lab)
        if best is None:
            raise ValueError("DBSCAN no encontro al menos 2 clusters en y")
        lab = best[1].copy()
        if -1 in lab:                               # el ruido se asigna al cluster mas cercano
            core = lab != -1
            nn = KNeighborsClassifier(1).fit(y[core], lab[core])
            lab[~core] = nn.predict(y[~core])
        self.eps_ = best[0][2]
        return lab

    def fit(self, y, X=None):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.method == "uniform":
            lo, hi = y.min(), y.max()
            self.edges_ = np.linspace(lo, hi, self.n_classes + 1)[1:-1]
            lab = np.digitize(y[:, 0], self.edges_)
        else:
            if self.method == "kmeans":
                lab = KMeans(self.n_classes, n_init=10, random_state=self.random_state).fit_predict(y)
            elif self.method == "hierarchical":
                lab = AgglomerativeClustering(self.n_classes, linkage="ward").fit_predict(y)
            elif self.method == "dbscan":
                lab = self._dbscan_labels(y)
            else:
                raise ValueError(f"Metodo desconocido {self.method!r}; opciones: {METHODS}")
            # reordenar clusters de menor a mayor irradiancia media: 0=baja ... K-1=alta
            ids = np.unique(lab)
            order = ids[np.argsort([y[lab == i].mean() for i in ids])]
            remap = {old: new for new, old in enumerate(order)}
            lab = np.array([remap[i] for i in lab])
            self.nn_ = KNeighborsClassifier(1).fit(y, lab)
        self.n_classes_ = len(np.unique(lab))
        self.classes_ = np.arange(self.n_classes_)
        # rango de irradiancia observado en train para cada clase (para leyendas)
        self.ranges_ = [(float(y[lab == c].min()), float(y[lab == c].max())) for c in self.classes_]
        self.counts_ = [int((lab == c).sum()) for c in self.classes_]
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.method == "uniform":
            return np.digitize(y[:, 0], self.edges_)
        return self.nn_.predict(y)

    def fit_transform(self, y, X=None):
        return self.fit(y).transform(y)

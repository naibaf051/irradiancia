"""
Kernel Artificial Neural Network (clasificador) - basado en KANN.py del repositorio (Mora Paz, UNIR).

La idea del original se conserva: aproximar el kernel con Nystroem y entrenar un MLP sobre el mapa de
caracteristicas.  Cambios respecto al original:
  * Nystroem se reemplaza por KernelNystroem (misma matematica, pero calcula la Gram vectorizada en
    lugar de llamar al kernel par a par);
  * el original ignoraba gamma/coef0 en 'hyperbolic' (fijaba 0.01 y 0.0) y coef0 en 'rquadratic'; aqui
    se respetan los hiperparametros recibidos;
  * se anade predict_proba (el original no transformaba X en esa llamada) y un error claro si el kernel
    es desconocido (antes fit devolvia None).
"""
import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
from .KernelUtilities import make_kernel, KERNEL_NAMES


class KernelNystroem(BaseEstimator, TransformerMixin):
    """Aproximacion de Nystroem para un kernel de KernelUtilities (Gram vectorizada)."""
    def __init__(self, kernel="rbf", degree=2, gamma=0.1, coef0=0.0, n_components=100, random_state=None):
        self.kernel, self.degree, self.gamma, self.coef0 = kernel, degree, gamma, coef0
        self.n_components, self.random_state = n_components, random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rnd = check_random_state(self.random_state)
        n = X.shape[0]
        m = min(self.n_components, n)
        idx = rnd.permutation(n)[:m]
        self.components_ = X[idx]
        self.kernel_ = make_kernel(self.kernel, self.degree, self.gamma, self.coef0)
        basis_K = self.kernel_(self.components_, self.components_)
        U, S, V = svd(basis_K)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        return self

    def transform(self, X):
        K = self.kernel_(np.asarray(X, dtype=float), self.components_)
        return K @ self.normalization_.T


class KANNC(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(100,), activation="identity",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=True,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                 kernel='rbf', degree=2, gamma=0.1, coef0=0.0):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        self.feature_map_nystroem = None
        self.isfit = False
        if self.kernel == "linear":
            super().fit(X, y)
            return self
        if self.kernel not in KERNEL_NAMES:
            raise ValueError(f"Kernel desconocido: {self.kernel!r}. Opciones: linear, {KERNEL_NAMES}")
        self.feature_map_nystroem = KernelNystroem(self.kernel, self.degree, self.gamma, self.coef0,
                                                   random_state=self.random_state)
        super().fit(self.feature_map_nystroem.fit_transform(X), y)
        self.isfit = True
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def mtransform(self, X):
        if self.feature_map_nystroem is not None and self.isfit:
            return self.feature_map_nystroem.transform(X)
        return X

    def predict(self, X):
        return super().predict(self.mtransform(X))

    def predict_proba(self, X):
        return super().predict_proba(self.mtransform(X))

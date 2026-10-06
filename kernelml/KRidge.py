"""
KRidgeClassifier: RidgeClassifier + truco kernel.

RidgeClassifier codifica las clases como {-1, +1} (one-vs-rest), resuelve una regresion ridge
  min ||Y - Xw||^2 + alpha ||w||^2
y predice con argmax de la salida.  En la forma dual, w = X^T a con a = (X X^T + alpha I)^-1 Y, asi que
solo se necesitan productos punto X X^T.  Sustituyendolos por una matriz de Gram K(X, X) de cualquier
funcion kernel se obtiene el clasificador kernel ridge:
  a = (K + alpha I)^-1 (Y - mean(Y))        f(x) = K(x, X_train) a + mean(Y)
"""
import numpy as np
from scipy import linalg
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, validate_data
from .KernelUtilities import gram_matrix, KERNEL_NAMES


class KRidgeClassifier(RidgeClassifier):
    def __init__(self, alpha=1.0, kernel="rbf", degree=2, gamma=0.1, coef0=0.0, class_weight=None):
        super().__init__(alpha=alpha, class_weight=class_weight)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def _gram(self, X, Y):
        if self.kernel == "linear":
            return pairwise_kernels(X, Y, metric="linear")
        if self.kernel in KERNEL_NAMES:
            return gram_matrix(self.kernel, X, Y, self.degree, self.gamma, self.coef0)
        raise ValueError(f"Kernel desconocido: {self.kernel!r}. Opciones: linear, {KERNEL_NAMES}")

    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y, accept_sparse=False, y_numeric=False, multi_output=False)
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y).astype(float)
        if Y.shape[1] == 1:                        # caso binario: LabelBinarizer devuelve 1 columna
            Y = np.hstack([-Y, Y])
        self.classes_ = self._label_binarizer.classes_
        self.X_fit_ = X
        self.y_mean_ = Y.mean(axis=0)
        K = self._gram(X, X)
        A = K + self.alpha * np.eye(K.shape[0])
        try:                                       # K puede no ser definida positiva (p.ej. hyperbolic)
            self.dual_coef_ = linalg.solve(A, Y - self.y_mean_, assume_a="sym")
        except (linalg.LinAlgError, ValueError):
            self.dual_coef_ = linalg.lstsq(A, Y - self.y_mean_)[0]
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["dual_coef_", "X_fit_"])
        X = validate_data(self, X, reset=False)
        scores = self._gram(X, self.X_fit_) @ self.dual_coef_ + self.y_mean_
        return scores[:, 1] if scores.shape[1] == 2 and len(self.classes_) == 2 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

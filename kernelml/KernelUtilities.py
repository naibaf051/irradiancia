"""
Funciones kernel (Belanche) - version vectorizada y serializable.

Las formulas son las del archivo original KernelUtilities (Mora Paz, UNIR, GPL) incluido en
Regressor.ipynb; aqui se calculan sobre la matriz completa con numpy (en el original se calculaba
par a par con dos bucles de Python) y los kernels son objetos de clase (no closures) para que los
modelos puedan guardarse con joblib/pickle.  Las funciones par a par originales se conservan
(PAIRWISE / pairwise_gram) y se usan en el notebook para verificar que ambas versiones coinciden.
"""
import numpy as np
import numpy.linalg as la

np.seterr(divide='ignore', invalid='ignore')

# ----------------------------------------------------------------------------------------------
# Version par a par (original) -- referencia para validar la vectorizada
# ----------------------------------------------------------------------------------------------
def polynomial(x, y, degree=3, gamma=0.01, coef0=0.0):
    return (gamma * np.dot(x, y) + 1) ** degree

def mrbf(x, y, degree=3, gamma=0.01, coef0=0.0):
    return np.exp(-np.sum(gamma * (x - y) ** degree))

def hyperbolic(x, y, degree=3, gamma=0.01, coef0=0.0):
    return np.tanh(gamma * np.dot(x, y) + coef0)

def triangle(x, y, degree=3, gamma=0.01, coef0=0.0):
    norm = la.norm(np.subtract(x, y))
    return 1 - norm / gamma if norm <= gamma else 0

def radial_basic(x, y, degree=3, gamma=0.01, coef0=0.0):
    return np.sum(np.exp(-gamma * ((x - y) ** 2))) ** degree

def rquadratic(x, y, degree=3, gamma=0.01, coef0=0.1):
    norm = la.norm(np.subtract(x, y))
    return 1 - (norm ** 2) / (norm ** 2 + coef0)

def canberra(x, y, degree=3, gamma=0.01, coef0=0.0):
    d = x.shape[0]
    r = gamma * np.abs(x - y) / (np.abs(x) + np.abs(y))
    return 1 - np.sum(r[~np.isnan(r)]) / d

def truncated(x, y, degree=3, gamma=0.01, coef0=0.0):
    d = x.shape[0]
    val = 1 - np.abs(x - y) / gamma
    return np.sum(val[val > 0]) / d

PAIRWISE = {"poly": polynomial, "rbf": mrbf, "hyperbolic": hyperbolic, "triangle": triangle,
            "radial_basic": radial_basic, "rquadratic": rquadratic, "can": canberra, "tru": truncated}

def pairwise_gram(name, X, Y, degree=2, gamma=0.01, coef0=0.0):
    """Gram matrix con el bucle original (lento) - solo para validacion."""
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i, j] = PAIRWISE[name](x, y, degree, gamma, coef0)
    return K

# ----------------------------------------------------------------------------------------------
# Version vectorizada
# ----------------------------------------------------------------------------------------------
def _diff(X, Y):
    return X[:, None, :] - Y[None, :, :]

def _sqdist(X, Y):
    return np.sum(_diff(X, Y) ** 2, axis=2)

def _resolve_gamma(gamma, X):
    if isinstance(gamma, str):               # 'auto' / 'scale' como en sklearn
        d = X.shape[1]
        return 1.0 / d if gamma == "auto" else 1.0 / (d * X.var() if X.var() > 0 else d)
    return gamma

def _K_poly(X, Y, degree, gamma, coef0):
    return (gamma * (X @ Y.T) + 1) ** degree

def _K_rbf(X, Y, degree, gamma, coef0):                  # RBF generalizado (beta=degree)
    return np.exp(-np.sum(gamma * _diff(X, Y) ** degree, axis=2))

def _K_hyperbolic(X, Y, degree, gamma, coef0):
    return np.tanh(gamma * (X @ Y.T) + coef0)

def _K_triangle(X, Y, degree, gamma, coef0):
    n = np.sqrt(_sqdist(X, Y))
    return np.where(n <= gamma, 1 - n / gamma, 0.0)

def _K_radial_basic(X, Y, degree, gamma, coef0):         # ANOVA
    return np.sum(np.exp(-gamma * _diff(X, Y) ** 2), axis=2) ** degree

def _K_rquadratic(X, Y, degree, gamma, coef0):
    n2 = _sqdist(X, Y)
    den = n2 + coef0
    return 1 - np.where(den == 0, 0.0, n2 / np.where(den == 0, 1.0, den))

def _K_can(X, Y, degree, gamma, coef0):
    num = gamma * np.abs(_diff(X, Y))
    den = np.abs(X)[:, None, :] + np.abs(Y)[None, :, :]
    r = num / den
    return 1 - np.nansum(r, axis=2) / X.shape[1]

def _K_tru(X, Y, degree, gamma, coef0):
    val = 1 - np.abs(_diff(X, Y)) / gamma
    return np.sum(np.where(val > 0, val, 0.0), axis=2) / X.shape[1]

_GRAM = {"poly": _K_poly, "rbf": _K_rbf, "hyperbolic": _K_hyperbolic, "triangle": _K_triangle,
         "radial_basic": _K_radial_basic, "rquadratic": _K_rquadratic, "can": _K_can, "tru": _K_tru}
KERNEL_NAMES = list(_GRAM)

def gram_matrix(name, X, Y, degree=2, gamma=0.01, coef0=0.0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if name not in _GRAM:
        raise ValueError(f"Kernel desconocido: {name!r}. Opciones: {KERNEL_NAMES}")
    return _GRAM[name](X, Y, degree, _resolve_gamma(gamma, X), coef0)

class GramKernel:
    """Kernel callable K(X, Y) -> Gram (n, m).  Es una clase (no closure) => picklable."""
    def __init__(self, name, degree=2, gamma=0.01, coef0=0.0):
        self.name, self.degree, self.gamma, self.coef0 = name, degree, gamma, coef0

    def __call__(self, X, Y):
        return gram_matrix(self.name, X, Y, self.degree, self.gamma, self.coef0)

    def __repr__(self):
        return f"GramKernel({self.name}, degree={self.degree}, gamma={self.gamma}, coef0={self.coef0})"

def make_kernel(name, degree=2, gamma=0.01, coef0=0.0):
    return GramKernel(name, degree, gamma, coef0)

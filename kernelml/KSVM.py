"""
Kernel Support Vector Machine (clasificador) - basado en KSVM.py del repositorio (Mora Paz, UNIR).

Cambios respecto al original (necesarios para sklearn 1.x y para poder guardar el modelo):
  * los kernels personalizados vienen de KernelUtilities.make_kernel (vectorizados y picklables);
  * si `kernel` ya es un callable (modelo clonado / reajustado) se ajusta directamente en vez de
    devolver None;
  * se conserva el nombre del kernel en `kernel_name_`;
  * gamma por defecto 'auto' (el valor 'auto_deprecated' ya no existe en sklearn).
"""
from sklearn.svm import SVC
from .KernelUtilities import make_kernel

_BUILTIN = ("linear", "poly", "rbf")


class KSVC(SVC):
    # 'linear', 'poly' y 'rbf' son los de sklearn; el resto de nombres son kernels de Belanche
    # ('mrbf' = RBF generalizado de Belanche).
    def __init__(self, C=1.0, kernel='rbf', degree=2, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None, a=2):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)
        self.a = a

    def fit(self, X, y, sample_weight=None):
        if callable(self.kernel):                       # ya convertido
            super().fit(X, y, sample_weight)
            return self
        self.kernel_name_ = self.kernel
        if self.kernel in _BUILTIN:
            super().fit(X, y, sample_weight)
            return self
        name = {"mrbf": "rbf"}.get(self.kernel, self.kernel)
        self.kernel = make_kernel(name, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        super().fit(X, y, sample_weight)
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

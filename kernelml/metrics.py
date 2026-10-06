"""Metricas de clasificacion multiclase: Accuracy, F1, AUC (OvR), MCC, matriz de confusion."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize


def class_scores(model, X):
    """Puntuacion continua por clase (n, K) del pipeline: predict_proba si existe, si no decision_function."""
    s = np.asarray(model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X))
    if s.ndim == 1:                                  # caso binario
        s = np.column_stack([-s, s])
    return s


def ovr_auc(y_true, scores, classes):
    """AUC macro one-vs-rest usando puntuaciones (no necesitan ser probabilidades)."""
    Y = label_binarize(y_true, classes=classes)
    if Y.shape[1] == 1:
        Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])
    aucs = []
    for k in range(Y.shape[1]):
        if Y[:, k].min() == Y[:, k].max():           # clase ausente en test: AUC no definido
            continue
        aucs.append(roc_auc_score(Y[:, k], scores[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_classifier(model, X, y_true, classes):
    y_pred = model.predict(X)
    scores = class_scores(model, X)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_ovr": ovr_auc(y_true, scores, classes),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }

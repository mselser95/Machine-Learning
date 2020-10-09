"""
Utilidades para uso en el curso.
"""
from itertools import product

import numpy as np
import sklearn.metrics as sk_mt


def get_allocations(n_assets):
    """
    Calcula combinaciones posibles de alocaciones entre n activos
    Parámetros
        n_assets : Cantidad de activos en portfolio

    Resultado
        generator
    """
    it = product(range(n_assets + 1), repeat=n_assets)

    return (np.array(e) / n_assets for e in it if sum(e) == n_assets)


def binary_classification_metrics(model, X_test, y_true):
    """
    Computa ROC AUC, Accuracy Score y Kolmogorov-Smirnov para un modelo de
    clasificación binaria dado.
    Parámetros
        model         : Modelo ya fitteado
        X_test        : Dataset con features
        y_true        : Etiquetras correctas para el dataset X_test

    Resultado
        tuple (ROC AUC, Accuracy, K-S)
    """
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    y_pred = model.predict(X_test)
    fpr, tpr, _ = sk_mt.roc_curve(y_true, y_score)

    roc_auc = sk_mt.auc(fpr, tpr)
    accuracy = sk_mt.accuracy_score(y_true, y_pred)
    ks = np.max(tpr - fpr)

    return roc_auc, accuracy, ks

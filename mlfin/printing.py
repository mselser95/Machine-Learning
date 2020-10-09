"""
Utilidades para impresión de resultados.
"""
import numpy as np
import sklearn.model_selection as sk_ms
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from .utils import binary_classification_metrics


def print_validation_results(estimator, X, y, random_state=None):
    """
    Imprime en pantalla resultados de validación.
    Parámetros
        estimator    : Estimador compatible con API scikit-learn
        X            : Valores variables independientes.
                       numpy.ndarray (n filas por k columnas)
        y            : Valores variable dependiente.
                       numpy.ndarray
        random_state : (opcional) Fija seed del generador de números
                       pseudoaleatorios
    """
    print(f'Testeando {type(estimator)}')

    np.random.seed(random_state)
    scores = sk_ms.cross_val_score(estimator, X, y, cv=5)

    if isinstance(estimator, KerasRegressor):
        total_mse = ((y - y.mean()) ** 2).mean()
        scores = 1 + scores / total_mse

    print('Resultado de Cross-Validation:')
    print('Scores  : [', ', '.join(f'{n:.2%}' for n in scores), ']', sep='')
    print(f'Mean    : {scores.mean():.2%}')
    print(f'Std Dev : {scores.std():.2%}')


def print_classification_metrics(model, X_test, y_true):
    """
    Computa e imprimer resumen con ROC AUC, Kolmogorov-Smirnov y Accuracy Score
    para un modelo de clasificación binaria dado.
    Parámetros
        model         : Modelo ya fitteado
        X_test        : Dataset con features
        y_true        : Etiquetras correctas para el dataset X_test
    """
    auc, acc, ks = binary_classification_metrics(model, X_test, y_true)

    print(f'Accuracy           = {acc:.2%}')
    print(f'ROC AUC            = {auc:.2%}')
    print(f'Kolmogorov-Smirnov = {ks:.4f}')

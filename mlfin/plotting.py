"""
Utilidades para visualización de resultados.
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk_mt


def plot_roc_curve(model, X_test, y_true):
    """
    Grafica curva ROC para un modelo de clasificación binaria dado.
    Parámetros
        model         : Modelo ya fitteado
        X_test        : Dataset con features
        y_true        : Etiquetras correctas para el dataset X_test
    """
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = sk_mt.roc_curve(y_true, y_score)
    roc_auc = sk_mt.auc(fpr, tpr)

    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'r', label=f'AUC = {roc_auc:.2%}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()


def plot_feature_importance(feature_importances, feature_names):
    """
    Grafica importancia de features
    IMPORTANTE: Ambos iterables deben estar correlacionados en ubicación
    Parámetros
        feature_importances : iterable con valores de importancia
        feature_names       : iterable con nombres de los features
    """
    _, ax = plt.subplots()
    order = np.argsort(feature_importances)[::-1]

    y_pos = np.arange(len(feature_names))

    ax.set_title('Importance of different features')
    ax.barh(y_pos, np.array(feature_importances)[order])

    ax.set_xlabel('Feature importance')
    ax.set_ylabel('Features')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(feature_names)[order])

    ax.invert_yaxis()
    plt.show()

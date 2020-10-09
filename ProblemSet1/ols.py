#
#   Author: Matias Selser
#
# %%
import numpy as np
from numpy.linalg import inv
from numpy import random

from mlfin import printing  as printing

from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin


# %%
class EstimadorOLS(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, coefs_number=1):
        self.fit_intercept = fit_intercept
        self.coefs_number = coefs_number
        self.constante = 0
        self.beta = []

    def fit(self, X, y):

        if self.coefs_number != X.shape[1]:
            return

        if self.fit_intercept == True:
            ones = np.ones((X.shape[0], 1))
            X = np.append(ones, X, axis=1)
            self.coeficientes = np.dot(np.dot(inv(np.dot(X.transpose(), X)), X.transpose()), y)
            self.constante = self.coeficientes[0]
            self.beta = self.coeficientes[1:]
        else:
            self.beta = np.dot(np.dot(inv(np.dot(X.transpose(), X)), X.transpose()), y)

    def predict(self, X):

        if self.fit_intercept == True:
            ones = np.ones((X.shape[0], 1))
            X = np.append(ones, X, axis=1)
            Y_Pred = np.dot(X, self.coeficientes)
        else:
            Y_Pred = np.dot(X, self.beta)

        return Y_Pred


# %%
if __name__ == '__main__':
    #   Generate X and Y datasets
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=1234)
    y = y + 5

    #   Create estimator
    myest = EstimadorOLS(fit_intercept=True, coefs_number=X.shape[1])

    #   Test model
    printing.print_validation_results(myest, X, y, random_state=1234)
    original_coefs = myest.beta

    #   Train model
    myest.fit(X, y)

    print("Fitted coefs:\nConstante: " + str(myest.constante) + "\n" + str(myest.beta) + "\n")
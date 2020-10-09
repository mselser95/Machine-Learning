"""
Funciones auxiliares
"""
import numpy as np


def get_state(prices, t, n):
    """
    Devuelve 'n-day state representation' finalizando en t.
    """
    d = t - n + 1
    block = prices[d:t + 1] if d >= 0 else -d * [prices[0]] + prices[0:t + 1]
    res = [block[i + 1] / block[i] - 1. for i in range(n - 1)]

    return np.array([res])

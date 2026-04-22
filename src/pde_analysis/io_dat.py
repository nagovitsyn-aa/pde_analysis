import numpy as np


def load_ab_dat(filepath):
    """
    Формат:
    x   |a|   |b|
    """

    data = np.loadtxt(filepath)

    x = data[:, 0]
    a = data[:, 1]
    b = data[:, 2]

    return {
        "x": x, 
        "a": a,
        "b": b,
    }


def load_Wa(filepath):
    data = np.loadtxt(filepath)

    return {
        "t": data[:, 0],
        "Wa": data[:, 1],
    }
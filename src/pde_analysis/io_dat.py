import numpy as np


def load_ab_dat(filepath):
    """
    Формат:
    x   |a|   |b|
    """
    data = np.loadtxt(filepath, skiprows=1)  # Пропускаем первую строку с заголовками

    x = data[:, 0]
    a = data[:, 1]
    b = data[:, 2]

    return {
        "x": x, 
        "a": a,
        "b": b,
    }


def load_Wa(filepath):
    data = np.loadtxt(filepath,skiprows=1)

    return {
        "t": data[:, 0],
        "Wa": data[:, 1],
    }

def load_Wa_Wb(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    
    # Базовая структура возвращаемых данных
    result = {
        "t": data[:, 0],
        "Wa": data[:, 1],
    }
    
    # Проверяем наличие третьей колонки
    if data.shape[1] >= 3:
        result["Wb"] = data[:, 2]
    
    return result
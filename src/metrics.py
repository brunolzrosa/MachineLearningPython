import numpy as np

def RMSE_per_column(pred: np.ndarray, target: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(pred - target), axis=0))

def RMSE_total(pred: np.ndarray, target: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(pred - target)))
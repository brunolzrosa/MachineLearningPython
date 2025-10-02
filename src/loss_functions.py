import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def __call__(self, pred: np.ndarray, target: np.ndarray, l2_sum: float = 0.0 , l2_lambda: float = 0.0) -> float:
        """Simple call to the loss function"""
        pass

    @abstractmethod
    def d(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Derivative of the loss function"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Name of loss function"""
        pass

class SSE(LossFunction):
    def __call__(self, pred: np.ndarray, target: np.ndarray, l2_sum: float = 0, l2_lambda: float = 0) -> float:
        return 0.5 * np.sum(np.square(pred - target)) + l2_lambda * l2_sum
    
    def d(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return pred - target
    
    def __str__(self) -> str:
        return "SSE"

class BinaryCrossEntropy(LossFunction):
    eps: float = 1e-16
    
    def __call__(self, pred: np.ndarray, target: np.ndarray, l2_sum: float = 0, l2_lambda: float = 0) -> float:
        return -np.sum((target * np.log(pred + self.eps)) + (1 - target)*(np.log(1 - pred + self.eps)), dtype=float) + l2_lambda * l2_sum
    
    def d(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (pred - target)
    
    def __str__(self) -> str:
        return "Binary Cross-Entropy"
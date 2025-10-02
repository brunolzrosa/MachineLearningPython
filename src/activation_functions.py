import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Simple call to the activation function"""
        pass

    @abstractmethod
    def d(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the activation function"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Name of activation function"""
        pass

class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def d(self, x: np.ndarray) -> np.ndarray:
        s = self.__call__(x)
        return s * (1 - s)
    
    def __str__(self) -> str:
        return "Sigmoid"

class Linear(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def d(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def __str__(self) -> str:
        return "Linear"

class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0)
    
    def d(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    def __str__(self) -> str:
        return "ReLU"
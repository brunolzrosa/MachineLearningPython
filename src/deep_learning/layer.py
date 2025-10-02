import numpy as np
from src.activation_functions import ActivationFunction

class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation: ActivationFunction) -> None:
        self.weights: np.ndarray = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
        self.biases: np.ndarray = np.zeros((1, output_dim))
        self.activation: ActivationFunction = activation
        self._z: np.ndarray | None = None
        self._a: np.ndarray | None = None

    @property
    def z(self) -> np.ndarray:
        if self._z is None:
            raise ValueError("Z is None")
        return self._z
    
        
    @property
    def a(self) -> np.ndarray:
        if self._a is None:
            raise ValueError("A is None")
        return self._a
    
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA: np.ndarray, prev_A: np.ndarray, learning_rate: float, l2_lambda: float, is_out: bool) -> np.ndarray:
        dZ: np.ndarray = dA if is_out else dA * self.activation.d(self.Z)
        dW: np.ndarray = np.dot(prev_A.T, dZ) + l2_lambda * self.weights
        dB: float = np.sum(dZ, axis=0, keepdims=True) + l2_lambda * self.biases
        dA_prev: np.ndarray = np.dot(dZ, self.weights.T)
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB
        return dA_prev

    def l2_norm(self) -> float:
        return np.sum(np.square(self.weights)) + np.sum(np.square(self.biases))
    
    def __str__(self) -> str:
        return f"Layer(In_dim: {self.weights.shape[0]}; Out_dim: {self.weights.shape[1]}, Activation: {self.activation})"
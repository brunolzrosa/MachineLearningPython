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
            raise ValueError("z is None")
        return self._z
    
    @z.setter
    def z(self, new_z: np.ndarray) -> None:
        self._z = new_z
    
        
    @property
    def a(self) -> np.ndarray:
        if self._a is None:
            raise ValueError("a is None")
        return self._a
    
    @a.setter
    def a(self, new_a: np.ndarray) -> None:
        self._a = new_a
    

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = np.dot(x, self.weights) + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backward(self, da: np.ndarray, prev_a: np.ndarray, learning_rate: float, l2_lambda: float, is_out: bool) -> np.ndarray:
        dz: np.ndarray = da if is_out else da * self.activation.d(self.z)
        dW: np.ndarray = np.dot(prev_a.T, dz) + l2_lambda * self.weights
        dB: float = np.sum(dz, axis=0, keepdims=True) + l2_lambda * self.biases
        da_prev: np.ndarray = np.dot(dz, self.weights.T)
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB
        return da_prev

    def l2_norm(self) -> float:
        return np.sum(np.square(self.weights)) + np.sum(np.square(self.biases))
    
    def __str__(self) -> str:
        return f"Layer(In_dim: {self.weights.shape[0]}; Out_dim: {self.weights.shape[1]}, Activation: {self.activation})"
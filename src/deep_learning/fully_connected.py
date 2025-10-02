import numpy as np
from src.activation_functions import ActivationFunction
from src.loss_functions import LossFunction
from src.deep_learning.layer import Layer

class NeuralNetworkFullyConnected:
    def __init__(self, l2_lambda: float = 0.0) -> None:
        self.layers: list[Layer] = []
        self.l2_lambda: float = l2_lambda
        self._x: np.ndarray | None = None
    
    @property
    def x(self) -> np.ndarray:
        if self._x is None:
            raise ValueError("x is None")
        return self._x
    
    @x.setter
    def x(self, new_x: np.ndarray) -> None:
        self._x = new_x
    
    @property
    def num_layers(self) -> int:
        return len(self.layers)
    
    def add_layer(self, input_dim: int, output_dim: int, activation: ActivationFunction) -> None:
        self.layers.append(Layer(input_dim, output_dim, activation))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.x = X
        a: np.ndarray = X
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backpropagate(self, pred: np.ndarray, target: np.ndarray, loss: LossFunction, learning_rate: float) -> None:
        da = loss.d(pred, target)
        for i in reversed(range(self.num_layers)):
            prev_a = self.layers[i-1].a if i > 0 else self.x
            da = self.layers[i].backward(da, prev_a, learning_rate, self.l2_lambda, (i == self.num_layers - 1))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def network_l2(self):
        return sum(layer.l2_norm() for layer in self.layers)
    
    def train_batch(self, X: np.ndarray, y: np.ndarray, 
                    loss_function: LossFunction, 
                    X_val: np.ndarray | None = None, 
                    y_val: np.ndarray | None = None, 
                    epochs: int = 50, learning_rate: float = 0.01,
                    verbose: bool = False
                    ) -> list[tuple[float, float | None]]: # train_error / validation_error
        errors: list[tuple[float, float | None]] = []
        for epoch in range(epochs):
            pred: np.ndarray = self.forward(X)
            self.backpropagate(pred, y, loss_function, learning_rate)
            
            # Errors
            l2: float = self.network_l2()
            loss_train: float = loss_function(pred, y, l2, self.l2_lambda)
            if (X_val is not None and y_val is not None):
                loss_val = loss_function(self.predict(X_val), y_val, l2, self.l2_lambda)
            else: loss_val = None
            errors.append((loss_train, loss_val))
            if verbose:
                if epoch % (((epochs - 1) // 10) + 1) == 0:
                    if (X_val is not None and y_val is not None):
                        print(f"Epoch {epoch}, Training Loss: {loss_train:.5f}, Validation Loss: {loss_val:.5f}")
                    else:
                        print(f"Epoch {epoch}, Training Loss: {loss_train:.5f}")
        return errors
    
    def train_sgd(self, X: np.ndarray, y: np.ndarray,
                    loss_function: LossFunction,
                    X_val: np.ndarray | None = None, 
                    y_val: np.ndarray | None = None,
                    epochs: int = 50, learning_rate: float = 0.01,
                    verbose: bool = False
                ) -> list[tuple[float, float | None]]: # train_error / validation_error
        errors: list[tuple[float, float | None]] = []
        N: int = X.shape[0]
        for epoch in range(epochs):
            for idx in range(N):
                x_i = X[idx:idx+1]
                y_i = y[idx:idx+1]
                pred = self.forward(x_i)
                self.backpropagate(pred, y_i, loss_function, learning_rate)
            
            # Errors
            l2: float = self.network_l2()
            loss_train: float = loss_function(self.predict(X), y, l2, self.l2_lambda)
            if (X_val is not None and y_val is not None):
                loss_val = loss_function(self.predict(X_val), y_val, l2, self.l2_lambda)
            else: loss_val = None
            errors.append((loss_train, loss_val))
            if verbose:
                if epoch % (((epochs - 1) // 10) + 1) == 0:
                    if (X_val is not None and y_val is not None):
                        print(f"Epoch {epoch}, Training Loss: {loss_train:.5f}, Validation Loss: {loss_val:.5f}")
                    else:
                        print(f"Epoch {epoch}, Training Loss: {loss_train:.5f}")
        return errors
    
    def __str__(self) -> str:
        full_desc = f"NeuralNetworkFullyConnected(lambda={self.l2_lambda}\n"
        for layer in self.layers:
            full_desc += f"    {layer}\n"
        full_desc += f")"
        return full_desc
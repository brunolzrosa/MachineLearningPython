from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.activation_functions import ActivationFunction, Sigmoid, Linear
from src.loss_functions import LossFunction
from src.deep_learning.fully_connected import NeuralNetworkFullyConnected

class NetworkFactory(ABC):
    @abstractmethod
    def activation(self) -> ActivationFunction:
        """Activation of the NN inner layers"""
        pass

    @abstractmethod
    def output_activation(self) -> ActivationFunction:
        """Activation of the NN outplut layer"""
        pass

    def build(self, num_layers: int,
                 neurons_per_layer: int,
                 input_dim: int,
                 output_dim: int,
                 l2_lambda: float = 0.0) -> NeuralNetworkFullyConnected:
        NN = NeuralNetworkFullyConnected(l2_lambda)
        if num_layers > 0:
            NN.add_layer(input_dim, neurons_per_layer, self.activation())
            for _ in range(num_layers - 1):
                NN.add_layer(neurons_per_layer, neurons_per_layer, self.activation())
            NN.add_layer(neurons_per_layer, output_dim, self.output_activation())
        else:
            NN.add_layer(input_dim, output_dim, self.output_activation())        # Simple Regression if only one layer
        return NN

class TrainMethod(Enum):
    SGD = "sgd"
    BATCH = "batch"

####################################
## FUNCTIONS TO PLOT THE TRAINING ##
####################################
def plot_errors(errors: list[tuple[float, float | None]], lamb: float, num_layers: int, neurons_per_layer: int):
    epochs = np.arange(1, len(errors) + 1)
    train_errors = np.array([error[0] for error in errors])
    val_errors = np.array([error[1] for error in errors])
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_errors, label='Training Error', color='b', linestyle='-', marker='o')
    if errors[0][1] is not None:
        plt.plot(epochs, val_errors, label='Validation Error', color='r', linestyle='--', marker='x')
    plt.title(f"Training vs Validation Errors\nLambda = {lamb}, Num_layers = {num_layers}, Neurons per Layer = {neurons_per_layer}", fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.show()

def plot_training(lambdas: list[float], 
                 num_layers: list[int], 
                 neurons_per_layers: list[int],
                 input_dim: int,
                 output_dim: int,
                 X: np.ndarray,
                 y: np.ndarray,
                 X_validation: np.ndarray | None,
                 y_validation: np.ndarray | None,
                 factory: NetworkFactory,
                 train_method: str,
                 loss: LossFunction,
                 epochs: int,
                 learning_rate: float,
                 verbose: bool = False
                ) -> list[tuple[float, NeuralNetworkFullyConnected, float, int, int]]: 
                # val error / net / lambda / num_layers / neurons_per_layer
    results: list[tuple[float, NeuralNetworkFullyConnected, float, int, int]] = []
    errors_training: list[tuple[float, float | None]] = []
    for lamb in lambdas:
        for num_layer in num_layers:
            for neurons_per_layer in neurons_per_layers:
                NN = factory.build(num_layer, neurons_per_layer, input_dim, output_dim, lamb)
                match train_method:
                    case TrainMethod.SGD:
                        if verbose: print("SGD")
                        errors_training = NN.train_sgd(X, y, loss, X_validation, y_validation, epochs, learning_rate, verbose)
                    case TrainMethod.BATCH:
                        if verbose: print("BATCH")
                        errors_training = NN.train_batch(X, y, loss, X_validation, y_validation, epochs, learning_rate, verbose)
                    case _ :
                        raise ValueError("Invalid method")
                plot_errors(errors_training, lamb, num_layer, neurons_per_layer)
                if (X_validation is not None and y_validation is not None):
                    val_error = loss(NN.predict(X_validation), y_validation, NN.network_l2(), lamb)
                else:
                    val_error = loss(NN.predict(X), y, NN.network_l2(), lamb)
                results.append((val_error, NN, lamb, num_layer, neurons_per_layer))
    return results

## Function to plot the error heatmaps
result_t = tuple[float, NeuralNetworkFullyConnected, float, int, int]

def plot_heatmaps(results: list[result_t]
                 ) -> None: # val error / net / lambda / num_layers / neurons_per_layer
    lambdas: np.ndarray = np.sort(np.unique([result[2] for result in results]))
    per_lambda: list[list[result_t]] = sorted([[result for result in results if result[2] == lamb] for lamb in lambdas])
    for resultlist in per_lambda:
        df: pd.DataFrame = pd.DataFrame(columns=["Error", "Layers", "Neurons"])
        for result in resultlist:
            df.loc[len(df)] = [result[0], result[3], result[4]]
        heatmap_data: pd.DataFrame = df.pivot(index="Layers", columns="Neurons", values="Error")
        plt.figure(figsize=(10, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Errors by Network Parameters (lambda={resultlist[0][2]})")
        plt.xlabel("Neurons per layer")
        plt.ylabel("# Hidden layers")
        plt.tight_layout()
        plt.show()



####################
## ACTUAL FACTORY ##
####################
class RegressionSigmoidNetworkBuilder(NetworkFactory):
    def activation(self) -> ActivationFunction:
        return Sigmoid()
    
    def output_activation(self) -> ActivationFunction:
        return Linear()

""" Example of use (with the train_test_split of sklearn)
# Reading the data
data = pd.read_csv("energy_efficiency.csv")
X: np.ndarray = data.drop(["Y1", "Y2"], axis=1).to_numpy()
y: np.ndarray = data[["Y1", "Y2"]].to_numpy()

# Spliting the data
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=12)
X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=(0.15/0.85), shuffle=True, random_state=12)

# Defining the parameters
lambdas: list[float] = [0, 0.001, 0.0001]
n_layers: list[int] = [1, 2, 3]
nperlayers: list[int] = [100, 150, 250]
NUM_EPOCHS: int = 50
ETA: float = 0.01
INPUT_DIM: int = X.shape[1]
OUTPUT_DIM: int = y.shape[1]

# Training
results = plot_training(lambdas, n_layers, nperlayers, INPUT_DIM, OUTPUT_DIM, X_train, y_train, X_val, y_val, RegressionNetworkBuilder(), TrainMethod.SGD, SSE(), NUM_EPOCHS, ETA)

"""
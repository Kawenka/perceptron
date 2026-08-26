from matrixlib import Matrix
import math

class Neuron:
    def __init__(self, num_features: int):
        """
        Initialize the logistic regression model parameters.

        Args:
            num_features (int): The number of input variables.
        """
        self.W = Matrix(num_features, 1)
        self.W.randomize(-1.0, 1.0)
        self.bias = 0.0

    def sigmoid(self, value: float):
        """
        Compute the sigmoid activation for a single scalar value.
        """
        return 1.0 / (1.0 + math.exp(-value))

    def forward (self, X: Matrix) -> Matrix:
        """
        Execute the forward propagation pass.
        """
        Z = (X * self.W) + self.bias
        activation = Z.map(self.sigmoid)
        return activation

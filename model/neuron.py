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
    
    def compute_loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        num_samples = y_true.rows
        epsilon = 1e-15
        loss_sum = 0.0

        for i in range (num_samples):
            y = y_true[i, 0]
            pred = y_pred[i, 0]
            
            pred = max(epsilon, min(1,0 - epsilon, pred));

            sample_loss = (y * math.log(pred)) + ((1.0 - y) * math.log(1.0 - pred))
            loss_sum += sample_loss
        return -(loss_sum / num_samples)


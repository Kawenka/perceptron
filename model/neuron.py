from matrixlib import Matrix

class Neuron:
    def __init__(self, num_features: int):
        """
        Initialize the logistic regression model parameters.

        Args:
            num_features (int): The number of input variables.
        """
        self.W = Matrix(num_features, 1)
        self.W.randomize(-1, 1)
        self.bias = 0.0

import numpy as np


class Layer:
    def __init__(self, size, activation=None):
        """
        Initialize a layer with only output size and activation function.
        The input size is determined by the network during initialization.
        Args:
            size: Number of neurons in this layer.
            activation: String representation of the activation function (e.g., 'linear' 'relu', 'sigmoid').
        """
        self.size = size
        self.activation = activation
        self.W = None
        self.b = None

    def initialize(self, input_size):
        """
        Initialize weights and biases once the input size is known.
        Args:
            input_dim: Number of input features (or neurons from the previous layer).
        """
        self.W = np.random.randn(self.size, input_size) * 0.01
        self.b = np.zeros((self.size, 1))

    def forward(self, x):
        """
        Forward pass through the layer.
        Args:
            x: Input data (or output from the previous layer).
        Returns:
            z: Linear output.
            a: Activated output.
        """
        z = np.dot(self.W, x) + self.b
        a = self.activation(z)
        return z, a

    def _activation_function(self):
        if self.activation == "relu":
            return lambda z: np.maximum(0, z)

        elif self.activation == "sigmoid":
            return lambda z: self._sigmoid(z)

        else:
            """
            self.activation is None, therefor default to linear
            """
            return lambda z: z

    def _activation_derivative(self):
        if self.activation == "relu":
            pass
            # return lambda z: np.maximum(0, z)

        elif self.activation == "sigmoid":
            pass
            # return lambda z: self._sigmoid(z)

        else:
            """
            self.activation is None, therefor default to linear
            """
            pass
            # return lambda z: z

    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))


class MultiLayerPerceptron:
    def __init__(self, layers):
        """
        Initialize a deep neural network
        Args:
            layers: A list of Layer classes.
        """
        pass

    def train(self):
        pass

    def _forward(self):
        pass

    def _backward(self):
        pass

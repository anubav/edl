import numpy as np


class Activation:
    """An activation function"""

    def __init__(self, name, activation, d_activation, vectorize=True):
        self.name = name
        self.activation = np.vectorize(activation) if vectorize else activation
        self.derivative = np.vectorize(d_activation) if vectorize else d_activation

    def __call__(self, inputs, D=False) -> np.array:
        return self.activation(inputs) if (not D) else self.derivative(inputs)


# Some Common Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    return 1 - (tanh(x) ** 2)


# Common Activations
IDENTITY = Activation('None', lambda x: x, lambda x: 1)
RELU = Activation('ReLU',
                  lambda x: x if (x > 0) else 0,
                  lambda x: 1 if (x > 0) else 0)
SIGMOID = Activation('Sigmoid', sigmoid, sigmoid_derivative)
TANH = Activation('Hyperbolic', tanh, tanh_derivative)

import numpy as np
from .activations import IDENTITY


class Dataset:
    """Target for data analysis by a neural network"""

    def __init__(self, inputs, targets) -> None:
        self.inputs = inputs
        self.targets = targets
        self.size = len(inputs)


class Layer:
    """A layer of neurons"""

    def __init__(self, width, bias=0, activation=IDENTITY, dropout=False):
        self.width = width
        self.inputs = np.empty(width)
        self.outputs = np.empty(width)
        self.deltas = np.empty(width)
        self.bias = bias
        self.activation = activation
        self.dropout = dropout

    def activate(self) -> None:
        """
        Pass input values through the layer's activation function
        (and dropout mask, if applicable)
        """
        self.outputs = self.activation(self.inputs)
        if self.dropout:
            self.dropout_mask = np.random.randint(2, self.width)
            self.outputs *= self.dropout_mask


class Network:
    """A feed-forward neural network"""

    def __init__(self, layers) -> None:
        self.layers = layers
        self.depth = len(layers)
        self.width = [layer.width for layer in layers]
        self.weights = [None for i in range(self.depth)]
        self.training_record = []

    def propagate(self, inputs) -> None:
        """
        Propagate inputs through the network
        (storing both layer inputs and outputs)
        """
        for i in range(self.depth):
            layer = self.layers[i]
            layer.inputs = inputs + layer.bias
            layer.activate()
            inputs = np.dot(layer.outputs, self.weights[i])

    def back_propagate(self, targets) -> None:
        """Backpropagate deltas through the network"""
        deltas = self.layers[-1].outputs - targets
        for i in reversed(range(self.depth)):
            layer = self.layers[i]
            layer.deltas = layer.activation(
                layer.inputs, D=True) * np.dot(deltas, self.weights[i].T)
            deltas = layer.deltas

    def update_network(self, inputs, targets, learning_rate) -> None:
        """Update network weights using the method of gradient descent"""
        self.propagate(inputs)
        self.back_propagate(targets)
        for i in range(self.depth - 1):
            self.weights[i] -= learning_rate * (np.dot(self.layers[i].outputs.T, self.layers[i + 1].deltas))

    def __call__(self, dataset, train=False, initialize=False, seed=1, learning_rate=1, stats={}):
        """Analyze dataset using the network or train network on dataset."""
        if isinstance(dataset, np.ndarray):  # Process single datum
            datum = dataset
            self.propagate(datum)
            return self.layers[-1].outputs.flatten()
        elif isinstance(dataset, Dataset):  # Process entire dataset
            record = {}  # Create a new training record
            for stat_name in stats.keys():
                record[stat_name] = []  # Initialize training record
            if initialize:
                self.training_record = []  # Reset training record
                for i in range(self.depth - 1):
                    self.weights[i] = (2 * seed) * np.random.rand(self.width[i], self.width[i + 1]) - seed
                self.weights[self.depth - 1] = np.identity(self.layers[-1].width)
            for inputs, targets in zip(dataset.inputs, dataset.targets):
                if train:
                    self.update_network(inputs, targets, learning_rate)
                else:
                    self.propagate(inputs)
                for stat_name, stat in stats.items():  # Update training record
                    value = stat(self, targets)
                    record[stat_name].append(value)
            # Store the record
            self.training_record.append(record)
        else:
            raise ValueError('Input is a neither a datapoint nor a dataset.')

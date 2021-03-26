import numpy as np
import pandas as pd
from .activations import Activation, IDENTITY


class Dataset:
    """Target for data analysis by a neural network"""
    def __init__(self, inputs, targets) -> None:
        self.inputs = inputs
        self.targets = targets
        self.statistics = {}

    def add_statistic(self, stat_name, stat) -> None:
        """Add a statistic to include in the analysis"""
        self.statistics[stat_name] = stat

    def analyze_datum(self, outputs, targets, record) -> None:
        for stat_name, stat in self.statistics.items():
            value = stat(outputs, targets, record)
            record[stat_name].append(value)


class Layer:
    """A layer of neurons"""
    def __init__(self, width, bias=0, activation=IDENTITY, dropout=False) -> None:
        self.width = width
        self.inputs = np.empty(width)
        self.outputs = np.empty(width)
        self.deltas = np.empty(width)
        self.bias = bias
        self.activation = activation
        self.dropout = dropout
            
    def activate(self) -> None:
        """Pass input values through the layer's activation function (and dropout mask, if applicable)"""
        self.outputs = self.activation(self.inputs)
        if self.dropout:
            self.dropout_mask = np.random.randint(2, self.width)
            self.outputs *= self.dropout_mask 


class Network:
    """A feed-forward neural network (or multi-layer perceptron)"""
    def __init__(self, layers) -> None:
        self.layers = layers
        self.depth = len(layers)
        self.width = [layer.width for layer in layers]
        self.weights = [None for i in range(self.depth)]
        self.training_record = None

    def propagate(self, inputs) -> None:
        """Propagate inputs through the network (storing both layer inputs and outputs)"""
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
            layer.deltas = layer.activation(layer.inputs, D=True) * np.dot(deltas, self.weights[i].T)
            deltas = layer.deltas

    def update_weights(self, inputs, targets, learning_rate) -> None:
        """Update network weights using the method of gradient descent"""
        self.propagate(inputs)
        self.back_propagate(targets)
        for i in range(self.depth - 1):
            self.weights[i] -= learning_rate * (np.dot(self.layers[i].outputs.T, self.layers[i + 1].deltas))

    def __call__(self, dataset, train=False, initialize=1, learning_rate=1):
        """Analyze dataset using the network or train network on dataset.""" 
        if isinstance(dataset, np.ndarray):
            datum = dataset
            self.propagate(datum)
            outputs = pd.DataFrame(self.layers[-1].outputs.flatten())
            outputs.columns = ['Output']
            return outputs
        elif isinstance(dataset, Dataset):
            record = {}
            for stat_name in dataset.statistics.keys():
                record[stat_name] = []
            data_generator = ((dataset.inputs[i], dataset.targets[i]) for i in range(len(dataset.inputs)))
            if train:
                for i in range(self.depth - 1):
                    self.weights[i] = (2 * initialize) * np.random.rand(self.width[i], self.width[i + 1]) - initialize
                self.weights[self.depth - 1] = np.identity(self.layers[-1].width)
            for inputs, targets in data_generator:
                if train:
                    self.update_weights(inputs, targets, learning_rate)
                else: 
                    self.propagate(inputs)
                outputs = self.layers[-1].outputs
                dataset.analyze_datum(outputs, targets, record)
            return pd.DataFrame.from_dict(record)
        else:
            raise ValueError('Input is a neither a datapoint nor a dataset.')
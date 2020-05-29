#! /usr/bin/env python3

"""
    This is a one-off single neuron. It is a silly example, just to get the gist.
    It takes training data (numpy arrays) in the form of
        A vector of inputs, X, which can be any length.
        A vector of actual values, y,
          which is the same size as X, and is either 0 or 1.
    A random weight vector, wts, is generated automatically.

    During training,
        the neuron will feed forwards to make predictions, ŷ.
        These are compared with the actual values of y.
        Then back propagate to adjust wts to the correct values

    During testing,
        the neuron will calculate the values of ŷ
        the accuracy of the neuron is the comparison of y with ŷ
"""

import numpy as np


class Neuron:
    def __init__(self, X_vector, y_vector):
        self.input = self.normalise(X_vector)
        self.y_true = y_vector  # The actual values: y.
        self.y_pred = np.zeros(self.input.size)  # Predicted values vector
        self.wts = np.random.rand(self.input.size, 1)
        self.bias = np.random.rand()

    def normalise(self, data):
        data = (data - data.mean()) / data.std()
        return data

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def train(self):
        for i in range(self.input.size):
            guess = np.dot(self.input[i], self.wts[i]) + self.bias
            y_hat = self.sigmoid(guess)  # ŷ = σ(X·w + b)


#! /usr/bin/env python3

"""
    This is a one-off single neuron. It is a silly example, just to get the gist.
    It takes training data (numpy arrays) in the form of
        A vector of inputs, x, which can be any length.
        A vector of actual values, y,
          which is the same size as x, and is either 0 or 1.
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
    def __init__(self, x_vector, y_scalar):
        # Import x adding x[0] = 1, as coefficient of bias term
        self.input = np.insert(x_vector, 0, 1)  # (object, position, value)

        # create weight vector with wts[0] = bias
        self.wts = np.random.rand(self.input.size, 1)

        self.y_true = y_scalar  # The actual value: y.
        self.learnrate = 0.001

    def normalise(self, data):
        data = (data - data.mean()) / data.std()
        return data

    def guess(self):
        return np.dot(self.input, self.wts)

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))


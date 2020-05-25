#! /usr/bin/env python3

"""
This is a one-off single neuron. It is a silly example, just to get the gist.
"""

import numpy as np


class Neuron:
    def __init__(self, X_vector, y_vector):
        self.input = X_vector
        self.y = y_vector  # The actual values
        self.wts = np.random.rand(self.input.size, 1)
        self.output = np.zeros(self.y.size)

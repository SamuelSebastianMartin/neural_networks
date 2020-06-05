#! /usr/bin/env python3

"""
    This is to provide data which can be used with various 
    neural network programs.
"""

import numpy as np


class Data:
    def __init__(self, size):
        self.size = size
        self.gradient = np.random.randint(-10, 11)
        self.intercept = np.random.randint(-5, 6)
        self.x = np.random.rand(self.size) * 10

    def error_added(self):
        """
        Alternative x values with a bit of wobble
        """
        self.y = self.intercept + self.gradient * self.x
        error = np.random.rand(self.size) / 10
        return self.x

    def linear_regres(self):
        """
        Returns linear function: y = m*x + c
        """
        self.y = self.intercept + self.gradient * self.x
        return self.x, self.y

    def linear_categ(self):
        """
        Returns 0 or 1, separated at y = m*x + c
        i.e. y - m*x > c
        """
        self.x2 = np.random.rand(self.size) * 10
        self.y = np.where(self.x2 - (self.x * self.gradient) > self.intercept, 1, 0)

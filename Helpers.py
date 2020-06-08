#! /usr/bin/env python3

"""
    This is to provide data which can be used with various
    neural network programs.

"""


import numpy as np


class Data:
    """
    Use:
        from Helpers import Data
        d = Data(20)  # generates 20 rows of data

        ## For regression data
        x, y = d.linear_regres()  # Give perfect straight line
        y = d.add_variance(y)  # To avoid perfect correlation.

        ## For categorisation data
        x1, x2, y = d.linear_categ()
    """
    def __init__(self, size):
        self.size = size
        self.gradient = np.random.randint(0.5, 4)
        self.intercept = np.random.randint(0, 4)
        self.x = np.random.rand(self.size) * 10

    def add_variance(self, vector):
        """
        Takes a vector, and adds a little imperfection
        so that all points do not lie exactly on the line.
        The added variance is normally distributed, but random.
        It seems more thematic to vary y rather than x
        """
        self.y = self.intercept + self.gradient * self.x
        error_vector = np.random.normal(size=vector.size)
        wobbly_vector = vector + error_vector
        return wobbly_vector

    def linear_regres(self):
        """
        Returns vectors x and y, corresponding to some
        linear function: y = m*x + c
        with random values selected for x, m & c
        """
        self.y = self.intercept + self.gradient * self.x
        return self.x, self.y

    def linear_categ(self):
        """
        Returns 2 independent variable vectors x1 and x2 plus
        a y vector of categorical data: 0 or 1. The y category
        is whether the point (x1, x2) lies above x1 = m*x2 + c
        i.e. x1 - m*x2 > c
        m and c are created randomly by __init__()
        """
        self.x2 = np.random.rand(self.size) * 10
        self.y = np.where(self.x2 - (self.x * self.gradient) > self.intercept, 1, 0)
        return self.input, self.x2, self.y

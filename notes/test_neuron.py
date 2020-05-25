#! /usr/bin/env python3

"""
A test class for classes/Neuron.py
"""
import unittest
from Classes import Neuron
import numpy as np


class TestNeuron(TestCase):

    def test_init(self):
        X = np.array([1, 2, 3])
        y = np.array([1, 0, 0])
        n = Neuron(X, y)
        self.assertEqual(self.X.size, n.input.size)

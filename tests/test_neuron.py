#! /usr/bin/env python3

from unittest import TestCase
import numpy as np
from Neuron import Neuron

class TestNeuron(TestCase):

    def test_init(self):
        X = np.array([1, 2, 3])
        y = np.array([0, 0, 1])
        n = Neuron(X, y)
        self.assertEqual(n.wts.size, 4)
        self.assertEqual(n.input.size, 4)
        self.assertEqual(n.input[0], 1)
        self.assertLess(n.learnrate, 1)

    def test_normalise(self):
        """ Note: normalised [1, 0] = [1, -1] """
        X = np.array([1, 0])
        y = np.array([0, 0])
        n = Neuron(X, y)
        norm = n.normalise(X)
        self.assertEqual(norm[0], 1)
        self.assertEqual(norm[1], -1)

    def test_guess(self):
        X = np.array([0, 0])
        y = np.array([0, 0])
        n = Neuron(X, y)
        bias = n.wts[0]
        self.assertEqual(n.guess(), bias)

    def test_sigmoid(self):
        """Note: exp(0) = 1"""
        X = np.array([1, 0])
        y = np.array([0, 0])
        n = Neuron(X, y)
        result = n.sigmoid(0)
        answer = 0.5
        self.assertEqual(result, answer)


if __name__ == '__main__':
    unittest.main()

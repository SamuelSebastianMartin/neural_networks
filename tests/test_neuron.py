#! /usr/bin/env python3

from unittest import TestCase
import numpy as np
from Neuron import Neuron

class TestNeuron(TestCase):

    def test_init(self):
        X = np.array([1, 2, 3])
        y = np.array([0, 0, 1])
        n = Neuron(X, y)
        self.assertEqual(n.input.size, 3)
        self.assertEqual(n.y_pred.size, 3)
        self.assertEqual(n.y_pred[0], 0)
        self.assertEqual(n.wts.size, 3)
        self.assertEqual(n.y_true.size, 3)

    def test_normalise(self):
        """ Note: normalised [1, 0] = [1, -1] """
        X = np.array([1, 0])
        y = np.array([0, 0])
        n = Neuron(X, y)
        self.assertEqual(n.input[0], 1)
        self.assertEqual(n.input[1], -1)

    def test_sigmoid(self):
        """Note: exp(0) = 1"""
        X = np.array([1, 0])
        y = np.array([0, 0])
        n = Neuron(X, y)
        result = n.sigmoid(0)
        answer = 0.5
        self.assertEqual(result, answer)

    def test_train(self):
        X = np.array([1, 0])
        y = np.array([0, 0])
        n = Neuron(X, y)
        n.train()
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()

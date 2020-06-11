#! /usr/bin/env python3

from unittest import TestCase
import numpy as np
from Neuron import Regression

class TestRegression(TestCase):

    def setUp(self):
        x_matrix = np.array([1,2,3,4]).reshape((2,2))
        y_vector = np.array([2, 2])
        self.r = Regression(x_matrix, y_vector)

    def test_init(self):
        # Check that 1's column is added to left of X.
        self.assertEqual(self.r.X.size, 6)
        self.assertEqual(self.r.X[0,0], 1)
        # Check that weight vector, w, is correct size.
        self.assertEqual(self.r.w.size, self.r.X.shape[1])
        print(self.r.X)
        print(self.r.w)
        print(self.r.X @ self.r.w.transpose())

    def test_predict(self):
        self.r.w = np.array([1, 1, 1,])
        h = self.r.predict()
        self.assertEqual(h[0], 4)
        self.assertEqual(h[1], 8)




# class TestNeuron(TestCase):
# 
#     def test_init(self):
#         X = np.array([1, 2, 3])
#         y = 0
#         n = Neuron(X, y)
#         self.assertEqual(n.wts.size, 4)
#         self.assertEqual(n.input.size, 4)
#         self.assertEqual(n.input[0], 1)
#         self.assertLess(n.learnrate, 1)
# 
#     def test_normalise(self):
#         """ Note: normalised [1, 0] = [1, -1] """
#         X = np.array([1, 0])
#         y = 0
#         n = Neuron(X, y)
#         norm = n.normalise(X)
#         self.assertEqual(norm[0], 1)
#         self.assertEqual(norm[1], -1)
# 
#     def test_predict(self):
#         X = np.array([0, 0])
#         y = 0
#         n = Neuron(X, y)
#         bias = n.wts[0]
#         self.assertLessEqual(n.predict(), 1)
#         self.assertGreaterEqual(n.predict(), 0)
# 
#     def test_sigmoid(self):
#         """Note: exp(0) = 1"""
#         X = np.array([1, 0])
#         y = 0
#         n = Neuron(X, y)
#         result = n.sigmoid(0)
#         answer = 0.5
#         self.assertEqual(result, answer)
# 
#     def test_cost(self):
#         X = np.array([1, 0])
#         y = 0
#         n = Neuron(X, y)
# #        self.assertLessEqual(n.cost.size, 1)
#         print(n.cost)
#         n.y_true = 1
#         n.y_pred = 1
#         self.assertEqual(n.find_cost(), 0)
#         n.y_true = 0
#         n.y_pred = 1
#         self.assertEqual(n.find_cost(), 0.5)

if __name__ == '__main__':
    unittest.main()

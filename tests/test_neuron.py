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
        # Check array sizes with (2,2) input
        self.assertEqual(self.r.y.shape, self.r.hypothesis.shape)
        self.assertEqual(self.r.w.size, self.r.X.shape[1])
        # Check array sizes with (6,2) input
        x_values = np.array([1,1,2,2,3,3,4,4,5,5,6,6])
        x_one = x_values.reshape((6, 2))
        y_one = np.array([1,2,3,4,5,6])
        r_one = Regression(x_one, y_one)
        self.assertEqual(r_one.m, y_one.size)  # No. of rows
        self.assertEqual(r_one.y.size, y_one.size)
        self.assertEqual(r_one.hypothesis.size, r_one.m)
        self.assertEqual(r_one.w.size, r_one.n)  # Weight size is good
        # Check array sizes with (2,6) input
        x_two = x_values.reshape((2,6))
        y_two = np.array([1,2])
        r_two = Regression(x_two, y_two)
        self.assertEqual(r_two.m, y_two.size)
        self.assertEqual(r_two.y.size, y_two.size)
        self.assertEqual(r_two.w.size, r_two.n)

    def test_predict(self):
        """
            Check that weights and features multiply correctly.
        """
        self.r.w = np.array([1, 1, 1,])
        self.r.predict()
        self.assertEqual(self.r.hypothesis[0], 4)
        self.assertEqual(self.r.hypothesis[1], 8)

    def test_cost(self):
        # Linear fn: y = x + 1
        x_matrix = np.array([1])
        y_matrix = np.array([2])
        r_three = Regression(x_matrix, y_matrix)
        r_three.w = np.array([[1, 1]])
        r_three.predict()
        r_three.caluculate_cost()
        self.assertEqual(r_three.cost, 0)


if __name__ == '__main__':
    unittest.main()

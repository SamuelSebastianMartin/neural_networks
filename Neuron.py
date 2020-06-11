#! /usr/bin/env python3


import numpy as np


class Regression:
    """
        For multi-variate linear regression

        Use:
            ***
    """
    def __init__(self, X_features, y_true_values):
        self.X = self.add_ones_column( X_features)
        self.y = self.configure_y(y_true_values)
        self.w = np.random.rand(1, self.X.shape[0] + 1)

    def add_ones_column(self, X_features):
        """
            Adds a column of 1's to the feature matrix.
            This will be multiplied by the first weight,
            thus making that first weight the bias expression.
        """
        X = np.ones((X_features.shape[0], X_features.shape[1] + 1))
        X[:,1:] = X_features
        return X

    def configure_y(self, labels_vector):
        """
            Checks the number of rows is the same in X and y, and
            returns y as a column vector
        """
        if labels_vector.size != self.X.shape[0]:
            raise Exception('X & y not conformable')
        else:
            y = labels_vector.reshape((self.X.shape[0], 1))
            return y

    def generate_weight_vector():
        """
            Generates random weights, one for each column in X.
            Note, the first weight is the bias and is always
            multiplied by the 1 in the first column of X.
        """
        weight_vector = np.random.rand((1, self.X.shape[1]))

    def predict(self):
        """
            For each row, calulate the predicted value:
                1*bias + x1*w1 + x2*w2 + ... + xn*wn
        """
        hypothesis = np.dot(self.X, self.w)
        return hypothesis

    def find_cost(self):
        cost = ((self.y_true - self.y_pred) **2) / 2
        return cost

    def update_wts(self):
        multiplier =  -(y_true - y_pred) * y_pred * (1 - y_pred)
        self.wts = self.wts + self.learnrate * multiplier * self.input


# class Neuron:
#     """
#         This is a one-off single neuron.
#         It is a silly example, just to get the gist.
#         It takes training data (numpy arrays) in the form of
#             A vector of inputs, x, which can be any length.
#             A vector of actual values, y,
#               which is the same size as x, and is either 0 or 1.
#         A random weight vector, wts, is generated automatically.
# 
#         During training,
#             the neuron will feed forwards to make predictions, ŷ.
#             These are compared with the actual values of y.
#             Then back propagate to adjust wts to the correct values
# 
#         During testing,
#             the neuron will calculate the values of ŷ
#             the accuracy of the neuron is the comparison of y with ŷ
#     """
#     def __init__(self, x_vector, y_scalar):
#         # Import x adding x[0] = 1, as coefficient of bias term
#         self.input = np.insert(x_vector, 0, 1)  # (object, position, value)
# 
#         # create weight vector with wts[0] = bias
#         self.wts = np.random.rand(self.input.size, 1)
# 
#         self.y_true = y_scalar  # The actual value: y.
#         self.learnrate = 0.001
#         self.y_pred = self.predict()
#         self.cost = self.find_cost()
# 
#     def predict(self):
#         guess = np.dot(self.input, self.wts)
#         y_predicted = self.sigmoid(guess)
#         return y_predicted
# 
#     def sigmoid(self, value):
#         return 1 / (1 + np.exp(-value))
# 
#     def find_cost(self):
#         cost = ((self.y_true - self.y_pred) **2) / 2
#         return cost
# 
#     def update_wts(self):
#         multiplier =  -(y_true - y_pred) * y_pred * (1 - y_pred)
#         self.wts = self.wts + self.learnrate * multiplier * self.input

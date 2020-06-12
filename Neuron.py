#! /usr/bin/env python3


import numpy as np


class Regression:
    """
        For multi-variate linear regression

        Use:
            ***
    """
    def __init__(self, X_features, y_true_values):
        # Fixed Values:
        if len(X_features.shape) == 1:  # Catch one dimensional X matrix
            X_features = X_features.reshape((1, X_features.shape[0]))
            print('Warning: X matrix is assumed to be one row of features')
            print('If it is really several rows of one feature, reshape it.')
        self.X = self.add_ones_column( X_features)  # n x m matrix
        self.m = self.X.shape[0]  # No. of data inputs (X:rows)
        self.n = self.X.shape[1]  # No. of features + bias (X:columns)
        self.y = self.configure_y(y_true_values)

        # Changing values:
        self.w = np.random.rand(1, self.n)
        self.hypothesis = np.zeros((self.m, 1))  # h = b + w1X1 + w2X2.
        self.cost_record = []  # For plotting cost for each epoch

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
        if labels_vector.size != self.m:
            raise Exception('X & y not conformable')
        else:
            y = labels_vector.reshape((self.m, 1))
            return y

    def generate_weight_vector():
        """
            Generates random weights, one for each column in X.
            Note, the first weight is the bias and is always
            multiplied by the 1 in the first column of X.
        """
        weight_vector = np.random.rand((1, self.n))

    def predict(self):
        """
            For each row, calulate the predicted value:
                1*bias + x1*w1 + x2*w2 + ... + xn*wn
        """
        self.hypothesis = np.dot(self.X, self.w.transpose())

    def caluculate_cost(self):
        """
            This is not necessary for funtioning, but gives
            a way to check that there is some convergence.
            Cost = J(w) = 1/2m (hypothesis - y)' (hypothesis - y)
        """
        error = self.hypothesis - self.y
        self.cost = np.dot(error.transpose(), error) / (2 * self.m)
        self.cost_record.append(self.cost)
        print(self.cost)  # Soon to be replaced by graph of cost_record

    def update_wts(self):
        pass

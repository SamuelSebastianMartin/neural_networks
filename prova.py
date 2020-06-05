#! /usr/bin/env python3

from Helpers import Data
from Neuron import Neuron

def get_data(data, train_size):
    data.linear_categ() #  to get x1, x2, y=category
    #  training data
    tr_x1 = data.x[ :train_size]
    tr_x2 = data.x2[ :train_size]
    tr_y = data.y[ :train_size]
    train_data = (tr_x1, tr_x2, tr_y)
    #  testing data
    te_x1 = data.x[train_size: ]
    te_x2 = data.x2[train_size: ]
    te_y = data.y[train_size: ]
    test_data = (te_x1, te_x2, te_y)

    return train_data, test_data


def main():
    data = Data(600)
    print(dir(data))
#    test_size = 500
#    train_data, test_data = get_data(data, test_size)
#    for i in range(test_size):
#        x_vector = [train_data[0][i], train_data[1][i]]
#        y_scalar = train_data[2][i]
#        data.input = x_vector
#        data.y_true = y_scalar
#        data.y_pred = data.predict()
#        data.cost = data.find_cost()
#        print(data.cost)
#        data.update_wts()


if __name__ == '__main__':
    main()

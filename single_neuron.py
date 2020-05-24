#! /usr/bin/env python3

"""
This is a one-off single neuron. It is a silly example, just to get the gist.
"""

from random import random

class Neuron:
    def __init__(self, train_data_size, test_data_size, increment):
        self.guess = 36.4 # The number the computer will try to find.
        self.trainsize = train_data_size
        self.testsize = test_data_size
        self.alpha = increment
        self.weight = 0.7
        self.bias = 0.5
        self.training_data = self.get_training_data()
        self.test_data = self.get_test_data()

    def get_training_data(self):
        """
        Generates a list of tupple pairs.
        The first element is a random number (0, 100) the second is 0 or 1,
        depending on whether the number is above or below the guess
        0 -> random number (point) is below the target number (guess)
        1 -> random number is above the target number.
        """
        training_data = []
        for n in range(self.trainsize):
            point = random() * 100
            if point < self.guess:
                datum = (point, 0)
            else:
                datum = (point, 1)
            training_data.append(datum)
        return training_data

    def get_test_data(self):
        """
        Generates a list of random numbers (0, 100).
        This is just used as a way to test the accuracy of the neuron.
        """
        test_data = []
        for n in range(self.testsize):
            point = random() * 100
            test_data.append(point)
        return test_data

    def train(self):
        """
        Makes a guess as to whether the number is too low (h<0)
        then does nothing if the guess is correct.
        If it is the guess based on h is too high,
        wt is reduced, and vice versa.
        """
        for datum in self.training_data:
            h = (datum[0] * self.weight) + self.bias

            if h < 0:  #  train() guesses 'low'...
                if datum[1] == 0:  # ... & train is correct.
                    continue
                elif datum[1] == 1:  # ...& train() is wrong
                    self.weight += self.alpha  # increase wt


            else:  # train() guesses 'high'...
                if datum[1] == 1:  # ... & train is correct.
                    continue
                elif datum[1] == 0:  # ...& train() is wrong
                    self.weight -= self.alpha  # decrease wt

    def make_guess(self):
        higher_list = []
        lower_list = []
        for test_datum in self.test_data:
            h = (test_datum * self.weight) + self.bias
            if h < 0:
                lower_list.append(test_datum)
            if h > 0:
                higher_list.append(test_datum)
        print(min(lower_list), max(higher_list))





n = Neuron(10000, 10000, 0.01)
n.train()
n.make_guess()

import numpy as np


class Sigmoid:
    def __call__(self, x):
        """
        Computes the logistics function
        :param x: numpy array - a vector for which the sigmoid should be computed
        :return: numpy array (vector) including output of the logistic function
        """
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        """
        Computes the derivative of the logistic function at a given array of points.
        :param x: vector of points at which the derivative values have to be assessed
        :return: vector of derivatives corresponding to each of the input points
        """
        sigmoid_x = self(x)
        return sigmoid_x * (1 - sigmoid_x)

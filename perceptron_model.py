import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod


class PerceptronModel(metaclass=ABCMeta):
    def __init__(self, learning_rate=0.8, error_tolerance=2e-2):
        """
        Initialize the parameters of the model
        :param learning_rate: The learning rate of the model
        :param iterations: The number of iterations over the data set during the training
        """
        self.learning_rate = learning_rate
        self.error_tolerance = error_tolerance
        self.w = None # The weights vector
        self.errors = None # The errors axis for plotting

    @property
    def iterations(self):
        return len(self.errors)

    def fit(self, features, result):
        """
        Train the model with training data having features `features` and labeled with `labels`
        :param features: The features of the data set + the bias field
        :param labels: The labels of the data set
        """
        self.errors = []

        # Initialize the weights vector with random values from 0 to 1, with the size of the features
        self.w = np.random.rand(features.shape[1], 1)
        while True:
            print("ITERATION {}".format(len(self.errors) + 1))
            predictions = self.get_predictions(features)
            error = self.adjust(features, result, predictions)
            print("ERROR: {}".format(error))
            self.errors.append(error)
            # Stopping condition
            if error < self.error_tolerance:
                print("TOOK {} ITERATIONS".format(len(self.errors)))
                break

    @abstractmethod
    def get_predictions(self, features):
        raise NotImplementedError()

    def adjust(self, features, result, predictions):
        """
        Adjust the weight vector based on perceptron algorithm
        :param features: The input features of the model
        :param labels: The labels of the data set
        :param predictions: The predicted values of the current iteration
        :return: The error resulting from this iteration
        """
        error = 0
        for i, pred in enumerate(predictions):
            error += self.get_error(pred[0], result[i])
            self.w -= self.learning_rate * (pred[0] - result[i]) * (features[i].reshape(features[i].shape[0], 1)) / len(predictions)
        return error / len(predictions)

    @abstractmethod
    def get_error(self, prediction, result):
        raise NotImplementedError()

    def plot_iterations_to_error(self):
        """
        Plot the error vs the number of iterations
        """
        if not self.errors:
            raise ValueError("Run fit")
        X = range(1, self.iterations + 1)  # X = [1, ..., iterations]
        Y = self.errors  # errors[i] = error after iteration i
        plt.plot(X, Y)
        plt.show()


def normalize(col):
    """
    Normalize a column in the data

    :param col: The column to normalize
    :return: Normalized column
    """
    # Take `abs` to handle negative values.
    return col / (np.abs(col).max())

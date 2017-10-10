import numpy as np
import pandas as pd
from perceptron_model import PerceptronModel, normalize


class LinearRegression(PerceptronModel):
    def get_predictions(self, features):
        """
        Get the predicted values based on the input features

        - Multiply features matrix (n, m) with weight matrix (m, 1)

        :param features: The input features of the whole data set
        :return: The predicted values of all the data set entries
        """
        return np.matmul(features, self.w)

    def get_error(self, prediction, result):
        return 0.5 * ((prediction - result) ** 2)

    def test(self, test_x, test_y):
        """

        :param  test_x: input testing data for testing the linear regression model
        :param  test_y: corresponding output for input testing data
        :return Mean Square Error
        """
        predictions = self.get_predictions(test_x)
        error = 0
        for i, pred in enumerate(predictions):
            error += self.get_error(pred[0], test_y[i])
        return error / len(predictions)


if __name__ == '__main__':
    all_data = pd.read_csv("AA.csv")[["DISTANCE", "ARRIVAL_TIME", "DEPARTURE_TIME"]].dropna().sample(frac=1)

    all_data["DISTANCE"] = normalize(all_data["DISTANCE"])
    all_data["DURATION"] = normalize(all_data["ARRIVAL_TIME"] - all_data["DEPARTURE_TIME"])

    length = all_data.shape[0]
    training_data, test_data = np.split(all_data, [int(0.8 * length)])

    train_X = pd.DataFrame(training_data["DISTANCE"])
    train_X["Bias"] = 1
    train_X = np.array(train_X)
    train_Y = np.array(training_data["DURATION"])

    test_X = pd.DataFrame(test_data["DISTANCE"])
    test_X["Bias"] = 1
    test_X = np.array(test_X)
    test_Y = np.array(test_data["DURATION"])

    regression = LinearRegression(learning_rate=0.4, error_tolerance=2e-2)
    regression.fit(train_X, train_Y)
    print(regression.test(test_X, test_Y))
    regression.plot_iterations_to_error()
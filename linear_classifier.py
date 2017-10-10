import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_model import PerceptronModel, normalize


class LinearClassifier(PerceptronModel):
    def get_predictions(self, features):
        """
        Get the predicted classes based on the input features

        - Multiply features matrix (n, m) with weight matrix (m, 1)
        - Saturate the result:
            res[i] = 1  : res[i] > 0
            res[i] = 0  : res[i] <= 0
        :param features: The input features of the whole data set
        :return: The predicted classes of all the data set entries
        """
        res = np.matmul(features, self.w)
        for i, val in enumerate(res):
            res[i] = 1 if val > 0 else 0
        return res

    def get_error(self, prediction, result):
        return abs(prediction - result)

    def plot_classification(self, w1i, w2i, feature1, feature2, labels):
        """
        Plot the decision boundary with respect to the given features
        :param w1i: Index of the first feature
        :param w2i: Idex of the second feature
        :param feature1: Data of feature 1
        :param feature2: Data of feature 2
        :param labels: The labels of the dataset
        """
        # Get the corresponding weight values
        w1 = self.w[w1i]
        w2 = self.w[w2i]

        # Get the bias weight value
        w0 = self.w[-1]

        # Plot line using two points
        x = [-20, 20]
        y = [(-w2 * i - w0) / w1 for i in x]
        plt.plot(x, y, 'r-')

        # Get indices of fraud and good transactions
        fraud = [i for i, val in enumerate(labels) if val]
        good = [i for i, val in enumerate(labels) if not val]

        # Plot good transactions as `*`
        plt.scatter([feature1[i] for i in good], [feature2[i] for i in good], marker='*')
        # Plot fraud transactions as `+`
        plt.scatter([feature1[i] for i in fraud], [feature2[i] for i in fraud], marker='+')

        plt.show()



if __name__ == '__main__':
    data = pd.read_csv("creditcard.csv")
    cols = ["V{}".format(i) for i in range(2, 12)]
    features = data[cols]
    features["V0"] = 1

    for col in cols:
        features[col] = normalize(features[col])

    features = np.array(features)
    labels = np.array(data["Class"])
    classifier = LinearClassifier(learning_rate=0.5, error_tolerance=1e-2)
    classifier.fit(features, labels)

    classifier.plot_classification(0, 9, np.array(data["V2"]), np.array(data["V11"]), labels)
    classifier.plot_iterations_to_error()

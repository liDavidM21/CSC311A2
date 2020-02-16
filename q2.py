from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.plot(X[:,i], y, 'ro')
    
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    biased = np.insert(X, 0, 1, axis=1)
    a = np.matmul(biased.transpose(), biased)
    b = np.matmul(biased.transpose(), Y)
    t = np.linalg.solve(a,b)
    return t


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
    # Fit regression model
    w = fit_regression(X, y)
    # Compute fitted values, MSE, etc.
    result_y = []
    for i in x_test:
        biased_i = np.insert(i, 0, 1)
        result_y.append(np.dot(biased_i, w))
    print(mean_squared_error(y_test, result_y))
if __name__ == "__main__":
    main()


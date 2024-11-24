from common import labeled_images, labeled_digits, autograder_images
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


def preprocess(images):
    # Reshapes and divides a given set of images into two training sets
    # and two test sets
    x = images.reshape(images.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x, labeled_digits, random_state=0, test_size=0.2)
    return x_train, x_test, y_train, y_test


def default_knn(x_train, y_train):
    # Trains a k-nearest neighbor classifier on the given training sets
    # with default parameters
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    return knn


def optimized_knn(x_train, y_train):
    # Trains a k-nearest neighbor classifier on the given training sets
    # with optimized parameters found through a grid search

    # Start grid search with number of neighbors between 1 and 29 (inclusive)
    grid = GridSearchCV(KNeighborsClassifier(),
                        param_grid={'n_neighbors': range(1, 30)},
                        n_jobs=-1, cv=5, refit=True, verbose=1)
    grid.fit(x_train, y_train)
    print(grid.best_params_)

    # Train optimized classifier
    knn_optimized = KNeighborsClassifier(n_neighbors=grid.best_params_.get('n_neighbors'))
    knn_optimized.fit(x_train, y_train)

    return knn_optimized


def accuracy(classifier, x_test, y_test):
    # Retrieves the accuracy score of the given trained classifier on
    # the given test sets
    y_pred = classifier.predict(x_test)
    return accuracy_score(y_test, y_pred)


def compare_default_knn_with_optimized_knn():
    # Trains a default k-nearest neighbor classifier and an optimized
    # one based on the labeled_images, then compares their accuracies
    x_train, x_test, y_train, y_test = preprocess(labeled_images)

    knn = default_knn(x_train, y_train)
    knn_accuracy = accuracy(knn, x_test, y_test)
    print('Base KNN accuracy: ', knn_accuracy)

    knn_optimized = optimized_knn(x_train, y_train)
    optimized_knn_accuracy = accuracy(knn_optimized, x_test, y_test)
    print('Optimized KNN accuracy: ', optimized_knn_accuracy)

    print('Difference (negative means default performs better): ',
          optimized_knn_accuracy - knn_accuracy)


def knn_autograder():
    # Trains a default k-nearest neighbor classifier on the
    # labeled_images data, estimates the labels for the
    # autograder_images, and places the results in autograder.txt
    x_train, x_test, y_train, y_test = preprocess(labeled_images)
    knn = default_knn(x_train, y_train)

    x = autograder_images.reshape(autograder_images.shape[0], -1)
    estimate = accuracy(knn, x_test, y_test)
    prediction = knn.predict(x)

    result = np.append(estimate, prediction)
    pd.DataFrame(result).to_csv("autograder.txt", index=False, header=False)

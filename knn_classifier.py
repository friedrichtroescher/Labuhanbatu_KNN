"""
K-Nearest Neighbor (KNN) Classifier Implementation

This module implements the KNN algorithm for classifying students into
'Superior' and 'Not Superior' categories based on their academic scores.

Formula used: Euclidean Distance
d(x,y) = sqrt(sum((x_i - y_i)^2))
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbor classifier implementation.

    Attributes:
        k (int): Number of nearest neighbors to consider
        X_train (numpy.ndarray): Training feature data
        y_train (numpy.ndarray): Training labels
    """

    def __init__(self, k=3):
        """
        Initialize the KNN classifier.

        Args:
            k (int): Number of neighbors to consider (default: 3)
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Store the training data.

        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points.

        Formula: d(x,y) = sqrt(sum((x_i - y_i)^2))

        Args:
            x1 (numpy.ndarray): First point
            x2 (numpy.ndarray): Second point

        Returns:
            float: Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_single(self, x):
        """
        Predict the class for a single instance.

        Args:
            x (numpy.ndarray): Feature vector to classify

        Returns:
            str: Predicted class label
        """
        # Calculate distances to all training samples
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self.euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))

        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]

        # Get the labels of k nearest neighbors
        k_nearest_labels = [label for _, label in k_nearest]

        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        """
        Predict classes for multiple instances.

        Args:
            X_test (numpy.ndarray): Test feature data

        Returns:
            numpy.ndarray: Predicted class labels
        """
        X_test = np.array(X_test)
        predictions = [self.predict_single(x) for x in X_test]
        return np.array(predictions)

    def get_params(self):
        """
        Get the parameters of the classifier.

        Returns:
            dict: Dictionary containing classifier parameters
        """
        return {'k': self.k}

    def set_params(self, **params):
        """
        Set the parameters of the classifier.

        Args:
            **params: Arbitrary keyword arguments
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

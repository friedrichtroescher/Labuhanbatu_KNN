"""
================================================================================
K-NEAREST NEIGHBOR (KNN) CLASSIFIER IMPLEMENTATION
================================================================================

LECTURE OVERVIEW:
This module implements the K-Nearest Neighbor (KNN) algorithm, a supervised
machine learning technique used for classification tasks. In this implementation,
we classify students into 'Superior' and 'Not Superior' categories based on
their academic scores (Knowledge, Skills, Attitude) and demographic data (Gender).

KEY CONCEPTS:
1. Instance-Based Learning: KNN is a "lazy learner" - it doesn't build an
   explicit model during training. Instead, it stores all training examples
   and makes decisions at prediction time.

2. Distance-Based Classification: KNN assumes that similar data points
   (based on distance metrics) belong to the same class.

3. Non-Parametric Method: KNN makes no assumptions about the underlying
   data distribution, making it flexible for various data patterns.

DISTANCE METRIC USED:
We use Euclidean Distance to measure similarity between data points:
    d(x,y) = sqrt(sum((x_i - y_i)^2))

    Where:
    - x and y are two data points (feature vectors)
    - x_i and y_i are individual feature values
    - The sum is taken over all features

HYPERPARAMETER:
- k: The number of nearest neighbors to consider when making predictions
  (In this implementation, default k=3 as per the research paper)

AUTHOR: Based on research by Nanda Fahrezi Munazhif et al.
================================================================================
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    ============================================================================
    K-NEAREST NEIGHBOR CLASSIFIER CLASS
    ============================================================================

    This class implements the complete KNN algorithm with the following stages:
    1. Training (fit): Store training data in memory
    2. Prediction (predict): Use stored data to classify new instances

    ATTRIBUTES:
        k (int):
            Number of nearest neighbors to consider for voting
            - Smaller k: More sensitive to noise, complex decision boundaries
            - Larger k: Smoother decision boundaries, less sensitive to noise
            - Odd numbers preferred for binary classification to avoid ties

        X_train (numpy.ndarray):
            Training feature matrix (n_samples × n_features)
            Each row represents one student with features:
            [Gender_Encoded, Knowledge, Skills, Attitude]

        y_train (numpy.ndarray):
            Training labels vector (n_samples,)
            Contains classification labels: 'Superior' or 'Not Superior'

    ALGORITHM WORKFLOW:
        Training Phase:   Store all training examples
        Prediction Phase: For each test point:
                         1. Calculate distances to all training points
                         2. Find k nearest neighbors
                         3. Perform majority voting
                         4. Return the most common class
    ============================================================================
    """

    def __init__(self, k=3):
        """
        ------------------------------------------------------------------------
        CONSTRUCTOR: Initialize the KNN Classifier
        ------------------------------------------------------------------------

        PURPOSE:
            Set up the classifier with the desired number of neighbors (k).
            This is called when creating a new KNNClassifier object.

        PARAMETERS:
            k (int): Number of neighbors to consider for classification
                    Default is 3 (as recommended in the research paper)

        LECTURE NOTE:
            The choice of k is crucial:
            - k=1: Classifier uses only the closest neighbor (high variance)
            - k=3: Balances local patterns with robustness (good default)
            - k=5,7,9: More stable but may miss local patterns

            For binary classification, odd values prevent voting ties.

        INITIALIZATION:
            - self.k: Store the hyperparameter
            - self.X_train: Will hold training features (initially None)
            - self.y_train: Will hold training labels (initially None)
        ------------------------------------------------------------------------
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        ------------------------------------------------------------------------
        TRAINING METHOD: Store Training Data
        ------------------------------------------------------------------------

        PURPOSE:
            In KNN, "training" simply means storing all training examples.
            Unlike other ML algorithms, KNN doesn't build a model or learn
            parameters. It's a "lazy learner" that defers computation until
            prediction time.

        PARAMETERS:
            X_train (numpy.ndarray or list):
                Training feature matrix, shape (n_samples, n_features)
                Example: [[1, 85, 90, 88], [0, 75, 80, 82], ...]

            y_train (numpy.ndarray or list):
                Training labels, shape (n_samples,)
                Example: ['Superior', 'Not Superior', 'Superior', ...]

        WHAT HAPPENS:
            1. Convert input data to NumPy arrays (for efficient computation)
            2. Store the entire training set in memory
            3. These will be used later during prediction

        LECTURE NOTE:
            KNN is called a "lazy learner" or "instance-based learner" because:
            - No model parameters are learned
            - No mathematical optimization occurs
            - Training is O(1) time complexity - just storing data
            - All computation happens during prediction (prediction is expensive!)

        TIME COMPLEXITY: O(1) - just storing references
        SPACE COMPLEXITY: O(n * d) where n=samples, d=features
        ------------------------------------------------------------------------
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def euclidean_distance(self, x1, x2):
        """
        ------------------------------------------------------------------------
        DISTANCE CALCULATION: Euclidean Distance
        ------------------------------------------------------------------------

        PURPOSE:
            Measure the "similarity" between two data points by calculating
            the straight-line distance between them in feature space.

        MATHEMATICAL FORMULA:
            d(x,y) = sqrt(sum((x_i - y_i)^2))

            For our 4D feature space (Gender, Knowledge, Skills, Attitude):
            d = sqrt((x₁-y₁)² + (x₂-y₂)² + (x₃-y₃)² + (x₄-y₄)²)

        PARAMETERS:
            x1 (numpy.ndarray): First data point (feature vector)
            x2 (numpy.ndarray): Second data point (feature vector)

        RETURNS:
            float: Distance between the two points (always non-negative)

        EXAMPLE:
            Student A: [1, 85, 90, 88]  (Man, Knowledge=85, Skills=90, Attitude=88)
            Student B: [0, 75, 80, 82]  (Woman, Knowledge=75, Skills=80, Attitude=82)

            Step 1: Calculate differences: [1-0, 85-75, 90-80, 88-82]
                                         = [1, 10, 10, 6]

            Step 2: Square each difference: [1, 100, 100, 36]

            Step 3: Sum: 1 + 100 + 100 + 36 = 237

            Step 4: Take square root: sqrt(237) ≈ 15.39

            Distance = 15.39

        LECTURE NOTE:
            Why Euclidean Distance?
            - Intuitive: Represents straight-line distance
            - Geometric interpretation: Length of the hypotenuse
            - Works well when all features are on similar scales
            - Other options: Manhattan, Minkowski, Cosine (for different use cases)

            IMPORTANT: Feature scaling matters! If one feature has much larger
            values than others, it will dominate the distance calculation.

        TIME COMPLEXITY: O(d) where d is the number of features
        ------------------------------------------------------------------------
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_single(self, x):
        """
        ------------------------------------------------------------------------
        SINGLE PREDICTION: Classify One Instance
        ------------------------------------------------------------------------

        PURPOSE:
            Predict the class label for a single test instance using the
            K-Nearest Neighbor algorithm with majority voting.

        PARAMETER:
            x (numpy.ndarray): Feature vector to classify
                             Example: [1, 85, 90, 88]

        RETURNS:
            str: Predicted class label ('Superior' or 'Not Superior')

        ALGORITHM STEPS:
        ----------------

        STEP 1: Calculate Distances
            - Compute distance from test point to every training point
            - Store pairs of (distance, label)
            - Example output: [(15.39, 'Superior'), (8.54, 'Not Superior'), ...]

        STEP 2: Sort by Distance
            - Arrange all distances in ascending order
            - Smallest distances = most similar students
            - This is where "nearest" neighbors are identified

        STEP 3: Select K Nearest Neighbors
            - Take the first k entries from sorted list
            - These are the k most similar training examples
            - Example with k=3: [(5.1, 'Superior'), (7.2, 'Superior'),
                                  (8.5, 'Not Superior')]

        STEP 4: Majority Voting
            - Extract labels from k nearest neighbors
            - Count occurrences of each class
            - Return the most common label (mode)
            - Example: 2 'Superior' + 1 'Not Superior' → Predict 'Superior'

        VISUALIZATION:
            Imagine a 4D space where each point is a student:
            - The test student is a new point
            - We find the 3 closest training students
            - We ask: "What class are most of these neighbors?"
            - That's our prediction!

        LECTURE NOTE:
            This is the heart of KNN! The assumption is:
            "Birds of a feather flock together" - similar students
            (in terms of features) should have similar classifications.

            The voting mechanism makes KNN robust to outliers - one
            misclassified neighbor won't drastically affect the result
            if k > 1.

        TIME COMPLEXITY:
            O(n * d) for distance calculation where n=training samples, d=features
            O(n log n) for sorting
            Overall: O(n * d + n log n) ≈ O(n log n) typically
        ------------------------------------------------------------------------
        """
        # STEP 1: Calculate distances to all training samples
        # ------------------------------------------------------
        # We iterate through every training example and compute how far
        # the test point is from each one. This creates our "distance map".
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self.euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))

        # STEP 2: Sort by distance and get k nearest neighbors
        # -----------------------------------------------------
        # Sorting allows us to identify which training points are closest.
        # We use a lambda function to sort by the first element (distance).
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]  # Slice to get only k neighbors

        # STEP 3: Get the labels of k nearest neighbors
        # ----------------------------------------------
        # Extract just the labels, discarding the distance values.
        # These are the "votes" in our majority voting system.
        k_nearest_labels = [label for _, label in k_nearest]

        # STEP 4: Return the most common label (majority vote)
        # -----------------------------------------------------
        # Counter is a dictionary that counts occurrences of each label.
        # most_common(1) returns the label with highest count.
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        """
        ------------------------------------------------------------------------
        BATCH PREDICTION: Classify Multiple Instances
        ------------------------------------------------------------------------

        PURPOSE:
            Predict class labels for multiple test instances efficiently.
            This is a convenience method that applies predict_single to
            each test instance.

        PARAMETER:
            X_test (numpy.ndarray or list):
                Test feature matrix, shape (n_test_samples, n_features)
                Example: [[1, 85, 90, 88], [0, 75, 80, 82], ...]

        RETURNS:
            numpy.ndarray:
                Array of predicted labels, shape (n_test_samples,)
                Example: ['Superior', 'Not Superior', 'Superior', ...]

        IMPLEMENTATION:
            Uses list comprehension to apply predict_single to each row
            of the test data, then converts the result to a NumPy array
            for consistency with scikit-learn conventions.

        LECTURE NOTE:
            In production systems, this method could be optimized using:
            - Vectorization (NumPy operations)
            - KD-Trees or Ball Trees for faster neighbor search
            - Parallel processing for independent predictions

            For educational purposes, we keep it simple and readable.

        TIME COMPLEXITY: O(n_test * n_train * d) - can be expensive!
        ------------------------------------------------------------------------
        """
        X_test = np.array(X_test)
        predictions = [self.predict_single(x) for x in X_test]
        return np.array(predictions)

    def get_params(self):
        """
        ------------------------------------------------------------------------
        GETTER: Retrieve Classifier Parameters
        ------------------------------------------------------------------------

        PURPOSE:
            Return the current hyperparameters of the classifier.
            Useful for model inspection, logging, and compatibility with
            scikit-learn's API conventions.

        RETURNS:
            dict: Dictionary containing 'k' parameter
                 Example: {'k': 3}

        USE CASE:
            When performing grid search or hyperparameter tuning, we need
            to query the current parameter values.
        ------------------------------------------------------------------------
        """
        return {'k': self.k}

    def set_params(self, **params):
        """
        ------------------------------------------------------------------------
        SETTER: Update Classifier Parameters
        ------------------------------------------------------------------------

        PURPOSE:
            Dynamically update the classifier's hyperparameters.
            This is useful for:
            - Hyperparameter tuning
            - Cross-validation with different k values
            - A/B testing different configurations

        PARAMETERS:
            **params: Arbitrary keyword arguments
                     Example: set_params(k=5) changes k to 5

        RETURNS:
            self: Returns the classifier object (allows method chaining)

        IMPLEMENTATION:
            Uses Python's setattr() to dynamically set attributes by name.
            This provides flexibility but requires careful usage.

        EXAMPLE USAGE:
            knn = KNNClassifier(k=3)
            knn.set_params(k=5)  # Now k=5

            # Method chaining:
            knn.set_params(k=7).fit(X_train, y_train).predict(X_test)

        LECTURE NOTE:
            This design pattern (get_params/set_params) is part of
            scikit-learn's Estimator API, making our classifier compatible
            with tools like GridSearchCV and Pipeline.
        ------------------------------------------------------------------------
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

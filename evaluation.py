"""
================================================================================
EVALUATION METRICS MODULE FOR CLASSIFICATION MODELS
================================================================================

LECTURE OVERVIEW:
This module implements fundamental evaluation metrics used to assess the
performance of binary classification models. Understanding these metrics is
crucial for determining how well our KNN classifier performs in distinguishing
between 'Superior' and 'Not Superior' students.

WHY DO WE NEED EVALUATION METRICS?
-----------------------------------
Simply building a classifier is not enough. We must answer:
- How accurate is the model?
- Does it correctly identify Superior students (Recall)?
- When it predicts Superior, is it usually correct (Precision)?
- What types of errors does it make?

THE CONFUSION MATRIX: Foundation of Classification Metrics
-----------------------------------------------------------
The confusion matrix is a 2×2 table that visualizes the four possible outcomes
of binary classification:

                    PREDICTED
                  Superior | Not Superior
    ACTUAL    +-----------+--------------+
    Superior  |    TP     |      FN      |  (True Positive & False Negative)
              +-----------+--------------+
    Not       |    FP     |      TN      |  (False Positive & True Negative)
    Superior  +-----------+--------------+

KEY TERMINOLOGY:
----------------
TP (True Positive):  Model correctly predicts Superior
                    Reality: Superior, Prediction: Superior ✓

TN (True Negative):  Model correctly predicts Not Superior
                    Reality: Not Superior, Prediction: Not Superior ✓

FP (False Positive): Model incorrectly predicts Superior (Type I Error)
                    Reality: Not Superior, Prediction: Superior ✗
                    Also called "False Alarm"

FN (False Negative): Model incorrectly predicts Not Superior (Type II Error)
                    Reality: Superior, Prediction: Not Superior ✗
                    Also called "Miss"

EVALUATION METRICS (Formulas from the research paper):
-------------------------------------------------------

1. ACCURACY = (TP + TN) / (TP + TN + FN + FP)
   - Measures overall correctness
   - Range: [0, 1] where 1 = perfect classification
   - Limitation: Can be misleading with imbalanced datasets

2. PRECISION = TP / (TP + FP)
   - "When the model predicts Superior, how often is it correct?"
   - Focuses on the quality of positive predictions
   - High precision = few false alarms
   - Critical when false positives are costly

3. RECALL = TP / (TP + FN)
   - "Of all actual Superior students, how many did we find?"
   - Also called Sensitivity or True Positive Rate
   - High recall = few missed Superior students
   - Critical when false negatives are costly

4. F1-SCORE = 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of Precision and Recall
   - Balances both metrics into a single score
   - Useful when you need to find optimal balance

PRACTICAL INTERPRETATION:
-------------------------
Example: Identifying Superior students
- High Precision: When we award "Superior" recognition, we're usually right
- High Recall: We successfully identify most Superior students
- Trade-off: Increasing one often decreases the other

AUTHOR: Based on research by Nanda Fahrezi Munazhif et al.
================================================================================
"""

import numpy as np


class ConfusionMatrix:
    """
    ============================================================================
    CONFUSION MATRIX CLASS - Binary Classification
    ============================================================================

    PURPOSE:
        Computes and stores the four fundamental outcomes of binary
        classification, which form the basis for all evaluation metrics.

    ATTRIBUTES:
        TP (int): True Positives
            Count of instances correctly classified as positive
            Example: Superior students correctly identified as Superior

        TN (int): True Negatives
            Count of instances correctly classified as negative
            Example: Not Superior students correctly identified as Not Superior

        FP (int): False Positives (Type I Error)
            Count of negative instances incorrectly classified as positive
            Example: Not Superior students incorrectly labeled as Superior

        FN (int): False Negatives (Type II Error)
            Count of positive instances incorrectly classified as negative
            Example: Superior students incorrectly labeled as Not Superior

        positive_label (str): The label considered as "positive" class
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels

    LECTURE NOTE:
        The confusion matrix is not just a counting tool - it reveals the
        nature of our model's errors. By examining which quadrant has more
        errors (FP vs FN), we can understand the model's bias and decide
        if adjustments are needed.
    ============================================================================
    """

    def __init__(self, y_true, y_pred, positive_label='Superior'):
        """
        ------------------------------------------------------------------------
        CONSTRUCTOR: Build Confusion Matrix from Predictions
        ------------------------------------------------------------------------

        PURPOSE:
            Initialize the confusion matrix by comparing true labels with
            predicted labels, counting each of the four possible outcomes.

        PARAMETERS:
            y_true (array-like):
                Ground truth labels (actual classifications)
                Example: ['Superior', 'Not Superior', 'Superior', ...]

            y_pred (array-like):
                Predicted labels (model's predictions)
                Example: ['Superior', 'Superior', 'Not Superior', ...]

            positive_label (str):
                Which label to treat as the "positive" class
                Default: 'Superior' (in our student classification context)

        WORKFLOW:
            1. Store the positive label designation
            2. Convert inputs to NumPy arrays for efficient processing
            3. Initialize counters (TP, TN, FP, FN) to zero
            4. Call _compute() to populate the matrix

        LECTURE NOTE:
            In binary classification, we must designate one class as "positive"
            and the other as "negative". This is somewhat arbitrary but affects
            how we interpret Precision and Recall.

            In this project:
            - Positive class: 'Superior' (what we're trying to identify)
            - Negative class: 'Not Superior' (the default/null hypothesis)

            This choice makes sense because identifying Superior students
            is our primary goal - they are the "target" of the classification.
        ------------------------------------------------------------------------
        """
        self.positive_label = positive_label
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

        # Initialize all counters to zero
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        # Compute the matrix values
        self._compute()

    def _compute(self):
        """
        ------------------------------------------------------------------------
        PRIVATE METHOD: Calculate Confusion Matrix Components
        ------------------------------------------------------------------------

        PURPOSE:
            Iterate through all predictions and count occurrences of each
            outcome type (TP, TN, FP, FN) by comparing true vs predicted labels.

        ALGORITHM:
            For each prediction-truth pair:
            - If both are positive → Increment TP (correct positive prediction)
            - If both are negative → Increment TN (correct negative prediction)
            - If true is negative but predicted positive → Increment FP (false alarm)
            - If true is positive but predicted negative → Increment FN (miss)

        IMPLEMENTATION DETAILS:
            We use zip() to pair corresponding elements from y_true and y_pred,
            then apply conditional logic to categorize each pair.

        LECTURE NOTE:
            This is where we "decode" our model's performance. Each comparison
            tells a story:
            - TP & TN: Model is working correctly
            - FP: Model is too "eager" - predicting Superior when it shouldn't
            - FN: Model is too "cautious" - missing actual Superior students

        EXAMPLE WALKTHROUGH:
            Given y_true = ['Superior', 'Not Superior', 'Superior']
                  y_pred = ['Superior', 'Superior', 'Not Superior']

            Iteration 1: true='Superior', pred='Superior'
                → Both positive → TP += 1 → TP = 1

            Iteration 2: true='Not Superior', pred='Superior'
                → True negative, Pred positive → FP += 1 → FP = 1

            Iteration 3: true='Superior', pred='Not Superior'
                → True positive, Pred negative → FN += 1 → FN = 1

            Result: TP=1, TN=0, FP=1, FN=1

        TIME COMPLEXITY: O(n) where n is the number of predictions
        ------------------------------------------------------------------------
        """
        for true, pred in zip(self.y_true, self.y_pred):
            # Case 1: True Positive - correctly identified Superior student
            if true == self.positive_label and pred == self.positive_label:
                self.TP += 1

            # Case 2: True Negative - correctly identified Not Superior student
            elif true != self.positive_label and pred != self.positive_label:
                self.TN += 1

            # Case 3: False Positive - Not Superior student labeled as Superior
            elif true != self.positive_label and pred == self.positive_label:
                self.FP += 1

            # Case 4: False Negative - Superior student labeled as Not Superior
            elif true == self.positive_label and pred != self.positive_label:
                self.FN += 1

    def get_matrix(self):
        """
        ------------------------------------------------------------------------
        GETTER: Retrieve Confusion Matrix Values
        ------------------------------------------------------------------------

        PURPOSE:
            Return the confusion matrix components as a dictionary for easy
            access by other functions (e.g., metric calculation functions).

        RETURNS:
            dict: Dictionary with keys 'TP', 'TN', 'FP', 'FN'
                 Example: {'TP': 15, 'TN': 8, 'FP': 2, 'FN': 3}

        USE CASE:
            Other functions (calculate_accuracy, calculate_precision, etc.)
            need access to these values to compute their respective metrics.
        ------------------------------------------------------------------------
        """
        return {
            'TP': self.TP,
            'TN': self.TN,
            'FP': self.FP,
            'FN': self.FN
        }

    def display(self):
        """
        ------------------------------------------------------------------------
        DISPLAY METHOD: Print Formatted Confusion Matrix
        ------------------------------------------------------------------------

        PURPOSE:
            Visualize the confusion matrix in a human-readable table format
            that clearly shows the relationship between actual and predicted
            classifications.

        OUTPUT FORMAT:
            A 2×2 table with:
            - Rows: Actual classes (Superior, Not Superior)
            - Columns: Predicted classes (Superior, Not Superior)
            - Cells: Counts of each outcome type

        LECTURE NOTE:
            Visual representation is crucial for understanding model behavior.
            When presenting results:
            - Large values on the main diagonal (TP, TN) = good performance
            - Large values on the off-diagonal (FP, FN) = errors
            - Comparing FP vs FN reveals which type of error is more common

        INTERPRETATION GUIDE:
            Perfect Model:        Biased Model:         Balanced Errors:
            TP=20  FN=0           TP=15  FN=5           TP=15  FN=3
            FP=0   TN=20          FP=1   TN=19          FP=2   TN=20
        ------------------------------------------------------------------------
        """
        print("\nConfusion Matrix:")
        print("=" * 50)
        print(f"{'':20} | Predicted Superior | Predicted Not Superior")
        print("-" * 50)
        print(f"{'Actual Superior':20} | {self.TP:^18} | {self.FN:^22}")
        print(f"{'Actual Not Superior':20} | {self.FP:^18} | {self.TN:^22}")
        print("=" * 50)


def calculate_accuracy(confusion_matrix):
    """
    ============================================================================
    METRIC 1: ACCURACY - Overall Correctness
    ============================================================================

    DEFINITION:
        Accuracy measures the proportion of correct predictions (both positive
        and negative) out of all predictions made.

    MATHEMATICAL FORMULA:
        Accuracy = (TP + TN) / (TP + TN + FN + FP)

        Numerator:   TP + TN = All correct predictions
        Denominator: TP + TN + FN + FP = All predictions (total samples)

    PARAMETER:
        confusion_matrix (ConfusionMatrix): Confusion matrix object containing
                                           TP, TN, FP, FN values

    RETURNS:
        float: Accuracy value in range [0, 1]
              - 0.0 = All predictions wrong (0% accuracy)
              - 1.0 = All predictions correct (100% accuracy)
              - 0.85 = 85% of predictions are correct

    INTERPRETATION:
        "What percentage of all predictions were correct?"

        Example: Accuracy = 0.90 (90%)
        → Out of 100 students, 90 were classified correctly

    WHEN TO USE:
        ✓ Balanced datasets (roughly equal Superior/Not Superior students)
        ✓ When both types of errors (FP and FN) are equally costly
        ✓ As a general measure of model performance

    LIMITATIONS:
        ✗ Misleading with imbalanced datasets
          Example: If 95% of students are "Not Superior", a model that
          always predicts "Not Superior" achieves 95% accuracy but is useless!

        ✗ Doesn't distinguish between types of errors
          A model with many FPs vs many FNs can have the same accuracy

    EDGE CASE HANDLING:
        If total predictions = 0, return 0.0 to avoid division by zero

    LECTURE NOTE:
        Accuracy is intuitive but not always sufficient. In real-world
        applications like medical diagnosis or fraud detection, we often
        care more about specific types of errors, making Precision and
        Recall more informative metrics.
    ============================================================================
    """
    cm = confusion_matrix.get_matrix()
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']

    # Handle edge case: no predictions made
    if total == 0:
        return 0.0

    # Calculate accuracy: correct predictions / total predictions
    return (cm['TP'] + cm['TN']) / total


def calculate_precision(confusion_matrix):
    """
    ============================================================================
    METRIC 2: PRECISION - Quality of Positive Predictions
    ============================================================================

    DEFINITION:
        Precision measures the proportion of correct positive predictions
        out of all positive predictions made by the model.

    MATHEMATICAL FORMULA:
        Precision = TP / (TP + FP)

        Numerator:   TP = Correct positive predictions
        Denominator: TP + FP = All positive predictions

    PARAMETER:
        confusion_matrix (ConfusionMatrix): Confusion matrix object

    RETURNS:
        float: Precision value in range [0, 1]
              - 0.0 = No correct positive predictions
              - 1.0 = All positive predictions are correct
              - 0.85 = 85% of positive predictions are correct

    INTERPRETATION:
        "When the model predicts 'Superior', how often is it correct?"

        Example: Precision = 0.89 (89%)
        → When we label a student as "Superior", we're right 89% of the time
        → 11% of our "Superior" predictions are false alarms (FP)

    FOCUS:
        Precision focuses on minimizing FALSE POSITIVES (Type I errors)

    WHEN TO USE:
        ✓ When false positives are costly or undesirable
        ✓ When you want to ensure predictions are trustworthy

    REAL-WORLD EXAMPLES:
        • Spam detection: High precision means few legitimate emails marked as spam
        • Medical testing: High precision means few healthy patients misdiagnosed
        • Student classification: High precision means few "not superior" students
          incorrectly awarded recognition

    TRADE-OFF:
        Improving precision often reduces recall (and vice versa)
        - To increase precision: Be more conservative → Fewer FPs, but more FNs
        - To increase recall: Be more liberal → Fewer FNs, but more FPs

    EDGE CASE HANDLING:
        If no positive predictions were made (TP + FP = 0), return 0.0
        This prevents division by zero and indicates the model never
        predicted the positive class.

    LECTURE NOTE:
        Precision answers: "Can we trust a positive prediction?"
        High precision is crucial when the cost of false alarms is high.
    ============================================================================
    """
    cm = confusion_matrix.get_matrix()
    denominator = cm['TP'] + cm['FP']

    # Handle edge case: no positive predictions made
    if denominator == 0:
        return 0.0

    # Calculate precision: correct positives / all predicted positives
    return cm['TP'] / denominator


def calculate_recall(confusion_matrix):
    """
    ============================================================================
    METRIC 3: RECALL - Completeness of Positive Detection
    ============================================================================

    DEFINITION:
        Recall (also called Sensitivity or True Positive Rate) measures the
        proportion of actual positive instances that were correctly identified.

    MATHEMATICAL FORMULA:
        Recall = TP / (TP + FN)

        Numerator:   TP = Correctly detected positives
        Denominator: TP + FN = All actual positives

    PARAMETER:
        confusion_matrix (ConfusionMatrix): Confusion matrix object

    RETURNS:
        float: Recall value in range [0, 1]
              - 0.0 = No actual positives were found
              - 1.0 = All actual positives were found
              - 0.92 = 92% of actual positives were found

    INTERPRETATION:
        "Of all actual 'Superior' students, how many did we successfully identify?"

        Example: Recall = 0.92 (92%)
        → We successfully identified 92% of all Superior students
        → We missed 8% of Superior students (FN)

    FOCUS:
        Recall focuses on minimizing FALSE NEGATIVES (Type II errors)

    WHEN TO USE:
        ✓ When false negatives are costly or undesirable
        ✓ When you want to ensure you don't miss positive cases

    REAL-WORLD EXAMPLES:
        • Disease screening: High recall ensures few sick patients are missed
        • Fraud detection: High recall catches most fraudulent transactions
        • Student classification: High recall ensures we identify most
          deserving Superior students (don't deny deserving students recognition)

    TRADE-OFF:
        Improving recall often reduces precision (and vice versa)
        - High recall (liberal model): Catches most positives, but many FPs
        - High precision (conservative model): Predictions trustworthy, but misses cases

        WHICH IS MORE IMPORTANT?
        - Medical screening: Recall (don't miss sick patients)
        - Spam filter: Precision (don't block legitimate emails)
        - Student awards: Balanced (fair and accurate recognition)

    EDGE CASE HANDLING:
        If there are no actual positive instances (TP + FN = 0), return 0.0
        This prevents division by zero and indicates no positive class exists
        in the dataset.

    LECTURE NOTE:
        Recall answers: "Did we find all the positive cases?"
        High recall is crucial when missing a positive case has serious consequences.
    ============================================================================
    """
    cm = confusion_matrix.get_matrix()
    denominator = cm['TP'] + cm['FN']

    # Handle edge case: no actual positive instances in dataset
    if denominator == 0:
        return 0.0

    # Calculate recall: detected positives / all actual positives
    return cm['TP'] / denominator


def calculate_f1_score(precision, recall):
    """
    ============================================================================
    METRIC 4: F1-SCORE - Harmonic Mean of Precision and Recall
    ============================================================================

    DEFINITION:
        F1-score is the harmonic mean of Precision and Recall, providing a
        single metric that balances both concerns. It's useful when you need
        a single number to compare models.

    MATHEMATICAL FORMULA:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Alternative form:
        F1 = 2 * TP / (2 * TP + FP + FN)

    WHY HARMONIC MEAN?
        Unlike arithmetic mean, harmonic mean severely penalizes extreme values.
        If either Precision or Recall is very low, F1 will also be low.

        Example comparison:
        Case 1: Precision=0.9, Recall=0.9
            Arithmetic mean: (0.9 + 0.9) / 2 = 0.9
            F1 (harmonic):   2 * 0.9 * 0.9 / 1.8 = 0.9

        Case 2: Precision=0.9, Recall=0.1 (imbalanced!)
            Arithmetic mean: (0.9 + 0.1) / 2 = 0.5 (misleading)
            F1 (harmonic):   2 * 0.9 * 0.1 / 1.0 = 0.18 (reflects poor balance)

    PARAMETERS:
        precision (float): Precision value [0, 1]
        recall (float): Recall value [0, 1]

    RETURNS:
        float: F1-score in range [0, 1]
              - 0.0 = Poor model (at least one metric is 0)
              - 1.0 = Perfect model (both precision and recall are 1)
              - Higher values indicate better balance

    INTERPRETATION:
        "How well does the model balance precision and recall?"

        Example: F1 = 0.85
        → Good balance between precision and recall
        → Both metrics are reasonably high

    WHEN TO USE:
        ✓ When you need a single metric to optimize
        ✓ When both precision and recall are important
        ✓ When comparing multiple models
        ✓ With imbalanced datasets (more robust than accuracy)

    LIMITATIONS:
        ✗ Doesn't distinguish which metric (precision/recall) is lower
        ✗ Treats precision and recall as equally important
           (Use F-beta score if one is more important)
        ✗ Doesn't account for true negatives (TN)

    EDGE CASE HANDLING:
        If both precision and recall are 0, return 0.0 to avoid division by zero

    LECTURE NOTE:
        F1-score is widely used in machine learning competitions and research
        papers because it provides a single number that summarizes model
        quality while being more informative than accuracy alone.

        Rule of thumb:
        - F1 > 0.9: Excellent
        - F1 > 0.8: Good
        - F1 > 0.7: Fair
        - F1 < 0.7: Needs improvement
    ============================================================================
    """
    # Handle edge case: both metrics are zero
    if precision + recall == 0:
        return 0.0

    # Calculate F1-score: harmonic mean of precision and recall
    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(y_true, y_pred, positive_label='Superior'):
    """
    ============================================================================
    COMPREHENSIVE MODEL EVALUATION FUNCTION
    ============================================================================

    PURPOSE:
        Perform complete evaluation of a binary classification model by
        computing all metrics (Accuracy, Precision, Recall, F1-Score) and
        displaying results in a formatted, easy-to-understand manner.

    PARAMETERS:
        y_true (array-like):
            Ground truth labels (actual classifications)
            Example: ['Superior', 'Not Superior', 'Superior', ...]

        y_pred (array-like):
            Predicted labels (model's predictions)
            Example: ['Superior', 'Superior', 'Not Superior', ...]

        positive_label (str):
            Which label to treat as the "positive" class
            Default: 'Superior'

    RETURNS:
        dict: Dictionary containing all evaluation results:
            {
                'confusion_matrix': {'TP': int, 'TN': int, 'FP': int, 'FN': int},
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float
            }

    WORKFLOW:
        STEP 1: Build confusion matrix from true and predicted labels
        STEP 2: Calculate all four metrics using the confusion matrix
        STEP 3: Display confusion matrix in tabular format
        STEP 4: Display all metrics with both decimal and percentage formats
        STEP 5: Return results dictionary for programmatic access

    OUTPUT FORMAT:
        The function prints:
        1. Confusion Matrix (2×2 table)
        2. Evaluation Metrics (with percentages)

    USE CASES:
        - After training a model, call this to see comprehensive results
        - Compare multiple models by comparing their returned dictionaries
        - Generate reports for stakeholders
        - Verify model meets performance requirements

    LECTURE NOTE:
        This function brings together all evaluation concepts into a single
        workflow. In practice, you'll call this once after making predictions
        to get a complete picture of model performance.

        When presenting results, focus on:
        - Confusion matrix: Show WHERE the model makes errors
        - Accuracy: Overall performance summary
        - Precision & Recall: Balance between false alarms and misses
        - F1-Score: Single number for model comparison

    EXAMPLE USAGE:
        >>> from knn_classifier import KNNClassifier
        >>> from evaluation import evaluate_model
        >>>
        >>> # Train model and make predictions
        >>> knn = KNNClassifier(k=3)
        >>> knn.fit(X_train, y_train)
        >>> y_pred = knn.predict(X_test)
        >>>
        >>> # Comprehensive evaluation
        >>> results = evaluate_model(y_test, y_pred)
        >>>
        >>> # Access specific metrics
        >>> print(f"Model accuracy: {results['accuracy']:.2%}")
    ============================================================================
    """
    # STEP 1: Build confusion matrix
    # -------------------------------
    # Creates ConfusionMatrix object which computes TP, TN, FP, FN
    cm = ConfusionMatrix(y_true, y_pred, positive_label)

    # STEP 2: Calculate all evaluation metrics
    # -----------------------------------------
    # Each function takes the confusion matrix and computes its metric
    accuracy = calculate_accuracy(cm)
    precision = calculate_precision(cm)
    recall = calculate_recall(cm)
    f1 = calculate_f1_score(precision, recall)

    # STEP 3: Display confusion matrix
    # ---------------------------------
    # Visualize the four outcome types in a 2×2 table
    cm.display()

    # STEP 4: Display all metrics
    # ---------------------------
    # Show each metric in both decimal (0.85) and percentage (85%) formats
    print("\nEvaluation Metrics:")
    print("=" * 50)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 50)

    # STEP 5: Return results dictionary
    # ----------------------------------
    # Allows programmatic access to all computed values
    return {
        'confusion_matrix': cm.get_matrix(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

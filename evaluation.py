"""
Evaluation Metrics Module

This module implements evaluation metrics for classification models:
- Confusion Matrix
- Accuracy
- Precision
- Recall

Formulas from the paper:
- Accuracy = (TP + TN) / (TP + TN + FN + FP)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
"""

import numpy as np


class ConfusionMatrix:
    """
    Confusion Matrix implementation for binary classification.

    Attributes:
        TP (int): True Positives
        TN (int): True Negatives
        FP (int): False Positives
        FN (int): False Negatives
    """

    def __init__(self, y_true, y_pred, positive_label='Superior'):
        """
        Initialize and compute confusion matrix.

        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            positive_label (str): Label to consider as positive class
        """
        self.positive_label = positive_label
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self._compute()

    def _compute(self):
        """
        Compute TP, TN, FP, FN values.
        """
        for true, pred in zip(self.y_true, self.y_pred):
            if true == self.positive_label and pred == self.positive_label:
                self.TP += 1
            elif true != self.positive_label and pred != self.positive_label:
                self.TN += 1
            elif true != self.positive_label and pred == self.positive_label:
                self.FP += 1
            elif true == self.positive_label and pred != self.positive_label:
                self.FN += 1

    def get_matrix(self):
        """
        Get confusion matrix values.

        Returns:
            dict: Dictionary with TP, TN, FP, FN values
        """
        return {
            'TP': self.TP,
            'TN': self.TN,
            'FP': self.FP,
            'FN': self.FN
        }

    def display(self):
        """
        Display confusion matrix in a readable format.
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
    Calculate accuracy from confusion matrix.

    Formula: Accuracy = (TP + TN) / (TP + TN + FN + FP)

    Args:
        confusion_matrix (ConfusionMatrix): Confusion matrix object

    Returns:
        float: Accuracy value (0-1)
    """
    cm = confusion_matrix.get_matrix()
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if total == 0:
        return 0.0
    return (cm['TP'] + cm['TN']) / total


def calculate_precision(confusion_matrix):
    """
    Calculate precision from confusion matrix.

    Formula: Precision = TP / (TP + FP)

    Args:
        confusion_matrix (ConfusionMatrix): Confusion matrix object

    Returns:
        float: Precision value (0-1)
    """
    cm = confusion_matrix.get_matrix()
    denominator = cm['TP'] + cm['FP']
    if denominator == 0:
        return 0.0
    return cm['TP'] / denominator


def calculate_recall(confusion_matrix):
    """
    Calculate recall from confusion matrix.

    Formula: Recall = TP / (TP + FN)

    Args:
        confusion_matrix (ConfusionMatrix): Confusion matrix object

    Returns:
        float: Recall value (0-1)
    """
    cm = confusion_matrix.get_matrix()
    denominator = cm['TP'] + cm['FN']
    if denominator == 0:
        return 0.0
    return cm['TP'] / denominator


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.

    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        precision (float): Precision value
        recall (float): Recall value

    Returns:
        float: F1 score (0-1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(y_true, y_pred, positive_label='Superior'):
    """
    Evaluate model and display all metrics.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        positive_label (str): Label to consider as positive class

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    cm = ConfusionMatrix(y_true, y_pred, positive_label)
    accuracy = calculate_accuracy(cm)
    precision = calculate_precision(cm)
    recall = calculate_recall(cm)
    f1 = calculate_f1_score(precision, recall)

    # Display results
    cm.display()
    print("\nEvaluation Metrics:")
    print("=" * 50)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 50)

    return {
        'confusion_matrix': cm.get_matrix(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

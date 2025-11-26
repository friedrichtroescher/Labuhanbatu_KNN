"""
Cross-Validation Analysis for KNN Classifier

This script performs k-fold cross-validation to get a more realistic
estimate of model performance across different train-test splits.
"""

import pandas as pd
import numpy as np
from knn_classifier import KNNClassifier
from evaluation import ConfusionMatrix, calculate_accuracy, calculate_precision, calculate_recall
from sklearn.model_selection import StratifiedKFold


def preprocess_data(data):
    """Preprocess the data for KNN classification."""
    data['Gender_Encoded'] = data['Gender'].map({'Man': 1, 'Woman': 0})
    X = data[['Gender_Encoded', 'Knowledge', 'Skills', 'Attitude']].values
    y = data['Category'].values
    return X, y


def cross_validate_knn(X, y, k_neighbors=3, n_folds=5):
    """
    Perform stratified k-fold cross-validation.

    Args:
        X (numpy.ndarray): Feature data
        y (numpy.ndarray): Labels
        k_neighbors (int): Number of neighbors for KNN
        n_folds (int): Number of folds for cross-validation

    Returns:
        dict: Dictionary with metrics for each fold and averages
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    print("\n" + "=" * 70)
    print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Number of neighbors (k): {k_neighbors}")
    print(f"Total samples: {len(X)}")
    print(f"Samples per fold (approx): {len(X) // n_folds}")
    print("=" * 70)

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Train KNN
        knn = KNNClassifier(k=k_neighbors)
        knn.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred_fold = knn.predict(X_test_fold)

        # Calculate metrics
        cm = ConfusionMatrix(y_test_fold, y_pred_fold, positive_label='Superior')
        accuracy = calculate_accuracy(cm)
        precision = calculate_precision(cm)
        recall = calculate_recall(cm)

        fold_results.append({
            'fold': fold_num,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.get_matrix(),
            'train_size': len(X_train_fold),
            'test_size': len(X_test_fold)
        })

        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(X_train_fold)} samples | Test: {len(X_test_fold)} samples")
        print(f"  Accuracy : {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall   : {recall*100:.2f}%")
        cm_dict = cm.get_matrix()
        print(f"  TP={cm_dict['TP']}, TN={cm_dict['TN']}, FP={cm_dict['FP']}, FN={cm_dict['FN']}")

    # Calculate averages
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])

    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_precision = np.std([r['precision'] for r in fold_results])
    std_recall = np.std([r['recall'] for r in fold_results])

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Average Accuracy  : {avg_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    print(f"Average Precision : {avg_precision*100:.2f}% ± {std_precision*100:.2f}%")
    print(f"Average Recall    : {avg_recall*100:.2f}% ± {std_recall*100:.2f}%")
    print("=" * 70)

    return {
        'fold_results': fold_results,
        'average': {
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall
        },
        'std': {
            'accuracy': std_accuracy,
            'precision': std_precision,
            'recall': std_recall
        }
    }


def test_different_k_values(X, y, k_values=[1, 3, 5, 7, 9], n_folds=5):
    """
    Test different k values to find optimal parameter.

    Args:
        X (numpy.ndarray): Feature data
        y (numpy.ndarray): Labels
        k_values (list): List of k values to test
        n_folds (int): Number of folds for cross-validation

    Returns:
        dict: Results for each k value
    """
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT K VALUES")
    print("=" * 70)

    results_by_k = {}

    for k in k_values:
        print(f"\n{'='*70}")
        print(f"Testing k={k}")
        print(f"{'='*70}")

        results = cross_validate_knn(X, y, k_neighbors=k, n_folds=n_folds)
        results_by_k[k] = results

    # Summary comparison
    print("\n" + "=" * 70)
    print("K-VALUE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'k':<5} {'Accuracy':<20} {'Precision':<20} {'Recall':<20}")
    print("-" * 70)

    for k in k_values:
        res = results_by_k[k]['average']
        std = results_by_k[k]['std']
        acc_str = f"{res['accuracy']*100:.2f}% ± {std['accuracy']*100:.2f}%"
        prec_str = f"{res['precision']*100:.2f}% ± {std['precision']*100:.2f}%"
        rec_str = f"{res['recall']*100:.2f}% ± {std['recall']*100:.2f}%"
        print(f"{k:<5} {acc_str:<20} {prec_str:<20} {rec_str:<20}")

    print("=" * 70)

    # Find best k
    best_k = max(k_values, key=lambda k: results_by_k[k]['average']['accuracy'])
    print(f"\nBest k value: {best_k}")
    print(f"Best average accuracy: {results_by_k[best_k]['average']['accuracy']*100:.2f}%")

    return results_by_k


def main():
    """Main function to run cross-validation analysis."""
    print("\n" + "=" * 70)
    print("KNN CROSS-VALIDATION ANALYSIS")
    print("MTs Swasta Islamiyah Kotapinang")
    print("=" * 70)

    # Load and preprocess data
    print("\nLoading data from student_data.csv...")
    data = pd.read_csv('student_data.csv')
    print(f"Loaded {len(data)} student records.")

    X, y = preprocess_data(data)

    # First, run cross-validation with k=3 (as per paper)
    print("\n" + "=" * 70)
    print("PART 1: Cross-validation with k=3 (paper's parameter)")
    print("=" * 70)
    results_k3 = cross_validate_knn(X, y, k_neighbors=3, n_folds=5)

    # Then test different k values
    print("\n" + "=" * 70)
    print("PART 2: Finding optimal k value")
    print("=" * 70)
    results_all_k = test_different_k_values(X, y, k_values=[1, 3, 5, 7, 9], n_folds=5)

    # Final comparison with paper
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 70)
    print("Paper Results (with their dataset):")
    print(f"  Accuracy  : 91.6%")
    print(f"  Precision : 89.2%")
    print(f"  Recall    : 92.5%")
    print("\nThis Implementation (5-fold CV, k=3):")
    avg = results_k3['average']
    std = results_k3['std']
    print(f"  Accuracy  : {avg['accuracy']*100:.2f}% ± {std['accuracy']*100:.2f}%")
    print(f"  Precision : {avg['precision']*100:.2f}% ± {std['precision']*100:.2f}%")
    print(f"  Recall    : {avg['recall']*100:.2f}% ± {std['recall']*100:.2f}%")
    print("=" * 70)

    print("\nNote: Cross-validation provides a more robust estimate of")
    print("performance by testing on multiple different train-test splits.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

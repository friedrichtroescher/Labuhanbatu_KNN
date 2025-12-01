"""
================================================================================
CROSS-VALIDATION ANALYSIS FOR KNN CLASSIFIER
================================================================================

LECTURE OVERVIEW:
This module implements k-fold cross-validation, an advanced technique for
obtaining more reliable estimates of model performance. Instead of a single
train-test split, we perform multiple evaluations to understand how well
the model generalizes.

WHY CROSS-VALIDATION?
---------------------
A single train-test split can be misleading:
- Results depend on which samples happen to be in the test set
- Small datasets have high variance in performance estimates
- We might get lucky (or unlucky) with the split

Cross-validation solves this by:
- Testing the model on multiple different test sets
- Averaging results across all folds
- Providing confidence intervals (mean ± std deviation)

K-FOLD CROSS-VALIDATION PROCESS:
---------------------------------
1. Split data into k equal-sized "folds" (typically k=5 or k=10)
2. For each fold:
   - Use that fold as test set
   - Use remaining (k-1) folds as training set
   - Train model and evaluate
3. Average results across all k folds
4. Report mean and standard deviation

EXAMPLE WITH k=5:
    Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
    Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
    Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
    Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
    Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]

STRATIFIED K-FOLD:
------------------
Regular k-fold might create imbalanced splits (e.g., all Superior students
in one fold). Stratified k-fold ensures each fold maintains the same class
distribution as the original dataset.

Example: If original data is 60% Superior, 40% Not Superior,
         each fold will also be ~60% Superior, ~40% Not Superior

HYPERPARAMETER TUNING:
----------------------
This module also implements grid search over different k values (number of
neighbors) to find the optimal hyperparameter for the KNN classifier.

AUTHOR: Based on research by Nanda Fahrezi Munazhif et al.
================================================================================
"""

import pandas as pd
import numpy as np
from knn_classifier import KNNClassifier
from evaluation import ConfusionMatrix, calculate_accuracy, calculate_precision, calculate_recall
from sklearn.model_selection import StratifiedKFold


def preprocess_data(data):
    """
    ============================================================================
    DATA PREPROCESSING FUNCTION
    ============================================================================

    PURPOSE:
        Convert raw CSV data into feature matrix (X) and label vector (y)
        suitable for machine learning algorithms.

    PARAMETER:
        data (pandas.DataFrame): Raw student data with columns:
            - Name: Student name (not used in modeling)
            - Gender: 'Man' or 'Woman'
            - Knowledge: Numeric score
            - Skills: Numeric score
            - Attitude: Numeric score
            - Category: 'Superior' or 'Not Superior' (target variable)

    RETURNS:
        tuple: (X, y) where:
            - X: numpy array of shape (n_students, 4)
                 [Gender_Encoded, Knowledge, Skills, Attitude]
            - y: numpy array of shape (n_students,)
                 ['Superior', 'Not Superior', ...]

    PREPROCESSING STEPS:
        1. Encode categorical variable (Gender) as numeric:
           'Man' → 1, 'Woman' → 0
        2. Select feature columns
        3. Extract target variable (Category)

    LECTURE NOTE:
        Machine learning algorithms require numeric input. We encode Gender
        as binary (0/1) to make it compatible with distance calculations.
        This is called "label encoding" or "binary encoding".

        Alternative encodings:
        - One-hot encoding: [Man: 1, 0] [Woman: 0, 1] (not needed for binary)
        - Target encoding: Use correlation with target (more advanced)
    ============================================================================
    """
    # Encode Gender as binary variable: Man=1, Woman=0
    data['Gender_Encoded'] = data['Gender'].map({'Man': 1, 'Woman': 0})

    # Create feature matrix: 4 features per student
    X = data[['Gender_Encoded', 'Knowledge', 'Skills', 'Attitude']].values

    # Extract target labels
    y = data['Category'].values

    return X, y


def cross_validate_knn(X, y, k_neighbors=3, n_folds=5):
    """
    ============================================================================
    STRATIFIED K-FOLD CROSS-VALIDATION FOR KNN
    ============================================================================

    PURPOSE:
        Perform k-fold cross-validation to obtain robust performance estimates
        by testing the model on multiple different train-test splits.

    PARAMETERS:
        X (numpy.ndarray):
            Feature matrix, shape (n_samples, n_features)
            Contains encoded Gender, Knowledge, Skills, Attitude

        y (numpy.ndarray):
            Labels vector, shape (n_samples,)
            Contains 'Superior' or 'Not Superior'

        k_neighbors (int):
            Number of neighbors for KNN classifier (default: 3)
            This is the 'k' in K-Nearest Neighbors

        n_folds (int):
            Number of folds for cross-validation (default: 5)
            Dataset will be split into n_folds equal parts

    RETURNS:
        dict: Comprehensive results dictionary containing:
            'fold_results': List of per-fold metrics
            'average': Mean metrics across all folds
            'std': Standard deviation of metrics

    ALGORITHM:
        FOR each fold i in 1 to n_folds:
            1. Split data: fold i becomes test set, others become train set
            2. Train KNN classifier on training set
            3. Predict on test set
            4. Calculate metrics (accuracy, precision, recall)
            5. Store results for this fold
        AFTER all folds:
            6. Calculate mean and std deviation across folds
            7. Display summary statistics

    WHY STRATIFIED?
        Maintains class distribution in each fold:
        - If 60% Superior in full dataset → ~60% Superior in each fold
        - Prevents biased results from imbalanced folds
        - Especially important for small datasets

    INTERPRETATION OF RESULTS:
        Mean: Expected performance on unseen data
        Std:  Variability/stability of the model
            - Low std: Model is consistent across different data splits
            - High std: Model is sensitive to training data composition

    LECTURE NOTE:
        Cross-validation is the gold standard for model evaluation because:
        - Uses all data for both training and testing
        - Reduces dependence on a single lucky/unlucky split
        - Provides confidence intervals (mean ± std)
        - Industry standard for model selection and comparison

        Common choices for n_folds:
        - k=5: Good balance between computation and reliability
        - k=10: More reliable but 2x slower
        - k=N (LOOCV): Most thorough but very slow for large datasets
    ============================================================================
    """
    # STEP 1: Initialize Stratified K-Fold splitter
    # -----------------------------------------------
    # Creates object that will generate k different train-test splits
    # shuffle=True: Randomize data before splitting (prevents order bias)
    # random_state=42: Ensures reproducible results
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # List to store results from each fold
    fold_results = []

    # STEP 2: Display cross-validation setup
    # ---------------------------------------
    print("\n" + "=" * 70)
    print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Number of neighbors (k): {k_neighbors}")
    print(f"Total samples: {len(X)}")
    print(f"Samples per fold (approx): {len(X) // n_folds}")
    print("=" * 70)

    # STEP 3: Perform cross-validation loop
    # --------------------------------------
    # For each fold, train on k-1 folds and test on the remaining fold
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Split data into training and testing sets for this fold
        # train_idx and test_idx are arrays of indices
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # SUBSTEP 3.1: Train KNN classifier
        # ----------------------------------
        # Create fresh KNN instance for this fold
        knn = KNNClassifier(k=k_neighbors)
        # Store training data (KNN's "training" process)
        knn.fit(X_train_fold, y_train_fold)

        # SUBSTEP 3.2: Make predictions
        # ------------------------------
        # Classify all samples in the test fold
        y_pred_fold = knn.predict(X_test_fold)

        # SUBSTEP 3.3: Calculate evaluation metrics
        # ------------------------------------------
        # Build confusion matrix and compute metrics
        cm = ConfusionMatrix(y_test_fold, y_pred_fold, positive_label='Superior')
        accuracy = calculate_accuracy(cm)
        precision = calculate_precision(cm)
        recall = calculate_recall(cm)

        # SUBSTEP 3.4: Store fold results
        # --------------------------------
        fold_results.append({
            'fold': fold_num,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.get_matrix(),
            'train_size': len(X_train_fold),
            'test_size': len(X_test_fold)
        })

        # SUBSTEP 3.5: Display fold results
        # ----------------------------------
        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(X_train_fold)} samples | Test: {len(X_test_fold)} samples")
        print(f"  Accuracy : {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall   : {recall*100:.2f}%")
        cm_dict = cm.get_matrix()
        print(f"  TP={cm_dict['TP']}, TN={cm_dict['TN']}, FP={cm_dict['FP']}, FN={cm_dict['FN']}")

    # STEP 4: Calculate aggregate statistics
    # ---------------------------------------
    # Compute mean (expected performance) across all folds
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])

    # Compute standard deviation (variability/consistency) across folds
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_precision = np.std([r['precision'] for r in fold_results])
    std_recall = np.std([r['recall'] for r in fold_results])

    # STEP 5: Display summary statistics
    # -----------------------------------
    # Report mean ± std for each metric
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Average Accuracy  : {avg_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    print(f"Average Precision : {avg_precision*100:.2f}% ± {std_precision*100:.2f}%")
    print(f"Average Recall    : {avg_recall*100:.2f}% ± {std_recall*100:.2f}%")
    print("=" * 70)

    # STEP 6: Return comprehensive results dictionary
    # ------------------------------------------------
    return {
        'fold_results': fold_results,  # Detailed per-fold results
        'average': {                    # Mean metrics
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall
        },
        'std': {                        # Standard deviation (variability)
            'accuracy': std_accuracy,
            'precision': std_precision,
            'recall': std_recall
        }
    }


def test_different_k_values(X, y, k_values=[1, 3, 5, 7, 9], n_folds=5):
    """
    ============================================================================
    HYPERPARAMETER TUNING: Grid Search for Optimal K
    ============================================================================

    PURPOSE:
        Test multiple values of k (number of neighbors) to find the optimal
        hyperparameter for the KNN classifier. This is known as grid search
        or hyperparameter tuning.

    PARAMETERS:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels vector
        k_values (list): List of k values to test
                        Default: [1, 3, 5, 7, 9]
        n_folds (int): Number of folds for cross-validation (default: 5)

    RETURNS:
        dict: Results for each k value tested, with structure:
            {
                1: {'fold_results': [...], 'average': {...}, 'std': {...}},
                3: {'fold_results': [...], 'average': {...}, 'std': {...}},
                ...
            }

    PROCESS:
        FOR each k value:
            1. Perform full k-fold cross-validation
            2. Store results (mean and std of metrics)
        AFTER testing all k values:
            3. Compare results in summary table
            4. Identify best k (highest average accuracy)

    WHY TEST DIFFERENT K VALUES?
        - k is a hyperparameter that affects model behavior
        - Small k (e.g., k=1): High variance, sensitive to noise
        - Large k (e.g., k=9): High bias, overly smooth decision boundary
        - Optimal k balances bias-variance tradeoff

    CHOOSING THE BEST K:
        Look for:
        ✓ Highest average accuracy
        ✓ Low standard deviation (consistent across folds)
        ✓ Good balance between precision and recall
        ✓ Computational efficiency (smaller k = faster prediction)

    LECTURE NOTE:
        This is an example of "model selection" - choosing the best
        configuration for your algorithm. In professional ML, this is
        extended to:
        - Testing multiple algorithms (KNN, Decision Trees, SVM, etc.)
        - Testing combinations of hyperparameters (grid search)
        - Using separate validation set or nested cross-validation
        - Considering multiple metrics simultaneously

        The k value chosen here becomes the "production" setting for
        deploying the model in real-world applications.
    ============================================================================
    """
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT K VALUES")
    print("=" * 70)

    # Dictionary to store results for each k value
    results_by_k = {}

    # STEP 1: Test each k value
    # --------------------------
    # Perform complete cross-validation for each candidate k
    for k in k_values:
        print(f"\n{'='*70}")
        print(f"Testing k={k}")
        print(f"{'='*70}")

        # Run cross-validation with this k value
        results = cross_validate_knn(X, y, k_neighbors=k, n_folds=n_folds)
        # Store results indexed by k
        results_by_k[k] = results

    # STEP 2: Display comparison summary
    # -----------------------------------
    # Create table comparing all k values side-by-side
    print("\n" + "=" * 70)
    print("K-VALUE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'k':<5} {'Accuracy':<20} {'Precision':<20} {'Recall':<20}")
    print("-" * 70)

    # Display each k value's performance with mean ± std
    for k in k_values:
        res = results_by_k[k]['average']  # Mean metrics
        std = results_by_k[k]['std']      # Standard deviations
        acc_str = f"{res['accuracy']*100:.2f}% ± {std['accuracy']*100:.2f}%"
        prec_str = f"{res['precision']*100:.2f}% ± {std['precision']*100:.2f}%"
        rec_str = f"{res['recall']*100:.2f}% ± {std['recall']*100:.2f}%"
        print(f"{k:<5} {acc_str:<20} {prec_str:<20} {rec_str:<20}")

    print("=" * 70)

    # STEP 3: Identify best k value
    # ------------------------------
    # Select k with highest average accuracy
    best_k = max(k_values, key=lambda k: results_by_k[k]['average']['accuracy'])
    print(f"\nBest k value: {best_k}")
    print(f"Best average accuracy: {results_by_k[best_k]['average']['accuracy']*100:.2f}%")

    return results_by_k


def main():
    """
    ============================================================================
    MAIN EXECUTION FUNCTION: Complete Cross-Validation Analysis
    ============================================================================

    PURPOSE:
        Orchestrate the complete cross-validation workflow, including:
        1. Data loading and preprocessing
        2. Cross-validation with k=3 (paper's parameter)
        3. Hyperparameter tuning (testing multiple k values)
        4. Comparison with published research results

    WORKFLOW:
        PART 1: Load and prepare data
        PART 2: Evaluate with k=3 (replicate paper methodology)
        PART 3: Find optimal k value through grid search
        PART 4: Compare results with research paper

    LECTURE NOTE:
        This main function demonstrates a complete ML evaluation pipeline:
        - Start with data loading and exploration
        - Perform rigorous evaluation (cross-validation)
        - Tune hyperparameters systematically
        - Validate against published baselines
        - Report comprehensive results

        This workflow is typical for academic research and professional
        ML projects, ensuring reproducibility and thorough evaluation.
    ============================================================================
    """
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

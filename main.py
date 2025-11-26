"""
Main Script: K-Nearest Neighbor Student Classification

Implementation of the research paper:
"Implementation of the K-Nearest Neighbor (kNN) Method to Determine Outstanding Student Classes"
by Nanda Fahrezi Munazhif, Gomal Juni Yanris, Mila Nirmala Sari Hasibuan

This script demonstrates:
1. Loading student data from CSV
2. Preprocessing features (Knowledge, Skills, Attitude, Gender)
3. Training KNN classifier (k=3 as per paper)
4. Making predictions
5. Evaluating with Confusion Matrix, Accuracy, Precision, Recall
"""

import pandas as pd
import numpy as np
from knn_classifier import KNNClassifier
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Data settings
    'data_file': 'student_data.csv',
    'positive_label': 'Superior',
    'negative_label': 'Not Superior',

    # Model settings
    'k_neighbors': 3,
    'test_size': 0.05,
    'random_state': None,  # None = truly random, or set to integer for reproducibility

    # School information
    'school_name': 'MTs Swasta Islamiyah Kotapinang',
    'project_title': 'K-NEAREST NEIGHBOR STUDENT CLASSIFICATION',

    # Paper benchmark results
    'paper_accuracy': 91.6,
    'paper_precision': 89.2,
    'paper_recall': 92.5,

    # Display settings
    'separator_width': 60,
    'name_column_width': 30,
    'category_column_width': 15,
    'correct_symbol': '✓',
    'incorrect_symbol': '✗',
}
# ============================================================================


def load_data(file_path='student_data.csv'):
    """
    Load student data from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} student records.\n")
    return data


def preprocess_data(data):
    """
    Preprocess the data for KNN classification.

    Features used:
    - Gender (encoded: Man=1, Woman=0)
    - Knowledge (numeric score)
    - Skills (numeric score)
    - Attitude (numeric score)

    Args:
        data (pandas.DataFrame): Raw data

    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    # Encode gender: Man=1, Woman=0
    data['Gender_Encoded'] = data['Gender'].map({'Man': 1, 'Woman': 0})

    # Select features: Gender, Knowledge, Skills, Attitude
    X = data[['Gender_Encoded', 'Knowledge', 'Skills', 'Attitude']].values

    # Target variable
    y = data['Category'].values

    return X, y


def display_data_summary(data):
    """
    Display summary statistics of the dataset.

    Args:
        data (pandas.DataFrame): Student data
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    # Category distribution
    category_counts = data['Category'].value_counts()
    print("\nClass Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {category:15}: {count:2} students ({percentage:.1f}%)")

    # Gender distribution
    print("\nGender Distribution:")
    gender_counts = data['Gender'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {gender:15}: {count:2} students ({percentage:.1f}%)")

    # Score statistics
    print("\nScore Statistics:")
    for column in ['Knowledge', 'Skills', 'Attitude']:
        print(f"\n  {column}:")
        print(f"    Mean   : {data[column].mean():.2f}")
        print(f"    Std Dev: {data[column].std():.2f}")
        print(f"    Min    : {data[column].min()}")
        print(f"    Max    : {data[column].max()}")

    print("=" * 60)


def main():
    """
    Main function to run the KNN classification pipeline.
    """
    # Get config values
    cfg = CONFIG
    sep = "=" * cfg['separator_width']
    train_pct = int((1 - cfg['test_size']) * 100)
    test_pct = int(cfg['test_size'] * 100)

    # Header
    print(f"\n{sep}")
    print(cfg['project_title'])
    print(cfg['school_name'])
    print(sep)

    # Load data
    data = load_data(cfg['data_file'])

    # Display summary
    display_data_summary(data)

    # Preprocess data
    print("\nPreprocessing data...")
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    print(f"Splitting data into train ({train_pct}%) and test ({test_pct}%) sets...")
    # Keep track of original indices to map back to student names
    indices = np.arange(len(data))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=cfg['test_size'], random_state=cfg['random_state'], stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # Initialize and train KNN classifier
    print(f"\n{sep}")
    print("TRAINING KNN CLASSIFIER")
    print(sep)
    print(f"Number of neighbors (k): {cfg['k_neighbors']}")

    knn = KNNClassifier(k=cfg['k_neighbors'])
    knn.fit(X_train, y_train)
    print("Training completed!")

    # Make predictions
    print(f"\n{sep}")
    print("MAKING PREDICTIONS")
    print(sep)
    y_pred = knn.predict(X_test)
    print(f"Predictions made for {len(y_test)} test samples.")

    # Evaluate model
    print(f"\n{sep}")
    print("MODEL EVALUATION")
    print(sep)
    results = evaluate_model(y_test, y_pred, positive_label=cfg['positive_label'])

    # Compare with paper results
    print(f"\n{sep}")
    print("COMPARISON WITH PAPER RESULTS")
    print(sep)
    print("Paper Results (Confusion Matrix method):")
    print(f"  Accuracy  : {cfg['paper_accuracy']}%")
    print(f"  Precision : {cfg['paper_precision']}%")
    print(f"  Recall    : {cfg['paper_recall']}%")
    print("\nThis Implementation:")
    print(f"  Accuracy  : {results['accuracy']*100:.1f}%")
    print(f"  Precision : {results['precision']*100:.1f}%")
    print(f"  Recall    : {results['recall']*100:.1f}%")
    print(sep)

    # Display all test case predictions with student names
    print(f"\n{sep}")
    print(f"ALL {len(y_test)} TEST CASES WITH STUDENT NAMES")
    print(sep)

    name_width = cfg['name_column_width']
    cat_width = cfg['category_column_width']
    table_width = name_width + cat_width * 2 + 10

    print(f"{'Name':<{name_width}} {'Actual':<{cat_width}} {'Predicted':<{cat_width}} {'Correct':<10}")
    print("-" * table_width)
    for i in range(len(y_test)):
        original_idx = test_indices[i]
        name = data.iloc[original_idx]['Name']
        actual = y_test[i]
        predicted = y_pred[i]
        correct = cfg['correct_symbol'] if actual == predicted else cfg['incorrect_symbol']
        print(f"{name:<{name_width}} {actual:<{cat_width}} {predicted:<{cat_width}} {correct:<10}")

    print(f"\n{sep}")
    print("Classification completed successfully!")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()

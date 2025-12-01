"""
================================================================================
MAIN SCRIPT: K-NEAREST NEIGHBOR STUDENT CLASSIFICATION SYSTEM
================================================================================

RESEARCH PAPER IMPLEMENTATION:
"Implementation of the K-Nearest Neighbor (kNN) Method to Determine
Outstanding Student Classes"

AUTHORS: Nanda Fahrezi Munazhif, Gomal Juni Yanris, Mila Nirmala Sari Hasibuan
INSTITUTION: MTs Swasta Islamiyah Kotapinang

LECTURE OVERVIEW:
================================================================================
This script demonstrates a complete machine learning pipeline from data
loading to model evaluation. It implements the KNN classification algorithm
to identify "Superior" students based on academic and demographic features.

MACHINE LEARNING PIPELINE STAGES:
==================================

1. DATA LOADING
   - Read student records from CSV file
   - Understand the data structure and format

2. EXPLORATORY DATA ANALYSIS (EDA)
   - Examine class distribution (Superior vs Not Superior)
   - Analyze gender distribution
   - Calculate statistical summaries of scores

3. DATA PREPROCESSING
   - Encode categorical variables (Gender)
   - Extract features (X) and labels (y)
   - Prepare data in ML-compatible format

4. TRAIN-TEST SPLIT
   - Divide data into training and testing sets
   - Training set: Used to "teach" the model
   - Testing set: Used to evaluate model performance
   - Stratified split: Maintains class balance

5. MODEL TRAINING
   - Initialize KNN classifier with k=3
   - Fit model to training data
   - (For KNN, this means storing training examples)

6. PREDICTION
   - Apply trained model to test data
   - Generate predicted labels

7. MODEL EVALUATION
   - Compute confusion matrix
   - Calculate metrics: Accuracy, Precision, Recall, F1-Score
   - Compare with research paper results

8. RESULTS PRESENTATION
   - Display all test cases with predictions
   - Show student names alongside classifications
   - Highlight correct vs incorrect predictions

FEATURES USED FOR CLASSIFICATION:
==================================
1. Gender (Encoded): Man=1, Woman=0
2. Knowledge Score: Academic knowledge assessment
3. Skills Score: Practical skills assessment
4. Attitude Score: Behavioral and character assessment

TARGET VARIABLE:
===============
Category: 'Superior' or 'Not Superior'

KEY PARAMETERS:
===============
- k_neighbors = 3 (as per research paper)
- test_size = 5% (95% training, 5% testing)
- Stratified sampling ensures representative splits

EXPECTED OUTCOMES:
==================
Students will learn:
- Complete ML workflow implementation
- Data preprocessing techniques
- Train-test split methodology
- KNN algorithm application
- Comprehensive model evaluation
- Results interpretation and comparison

================================================================================
"""

import pandas as pd
import numpy as np
from knn_classifier import KNNClassifier
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split


# ============================================================================
# CONFIGURATION DICTIONARY
# ============================================================================
# Centralized configuration for all parameters used in the pipeline.
# This makes it easy to modify settings without changing code throughout
# the script. This is a best practice in software engineering.
# ============================================================================
CONFIG = {
    # DATA SETTINGS
    # -------------
    'data_file': 'student_data.csv',        # Input CSV file name
    'positive_label': 'Superior',            # Positive class label
    'negative_label': 'Not Superior',        # Negative class label

    # MODEL HYPERPARAMETERS
    # ---------------------
    'k_neighbors': 3,                        # Number of neighbors (as per paper)
    'test_size': 0.05,                       # 5% for testing, 95% for training
    'random_state': None,                    # None = random, int = reproducible

    # INSTITUTION INFORMATION
    # -----------------------
    'school_name': 'MTs Swasta Islamiyah Kotapinang',
    'project_title': 'K-NEAREST NEIGHBOR STUDENT CLASSIFICATION',

    # RESEARCH PAPER BENCHMARK RESULTS (for comparison)
    # --------------------------------------------------
    'paper_accuracy': 91.6,                  # Published accuracy (%)
    'paper_precision': 89.2,                 # Published precision (%)
    'paper_recall': 92.5,                    # Published recall (%)

    # DISPLAY FORMATTING
    # ------------------
    'separator_width': 60,                   # Width of separator lines
    'name_column_width': 30,                 # Width for name column
    'category_column_width': 15,             # Width for category column
    'correct_symbol': '✓',                   # Symbol for correct prediction
    'incorrect_symbol': '✗',                 # Symbol for incorrect prediction
}
# ============================================================================


def load_data(file_path='student_data.csv'):
    """
    ============================================================================
    DATA LOADING FUNCTION
    ============================================================================

    PURPOSE:
        Read student data from a CSV file into a pandas DataFrame for
        analysis and modeling.

    PARAMETER:
        file_path (str): Relative or absolute path to the CSV file
                        Default: 'student_data.csv'

    RETURNS:
        pandas.DataFrame: Table containing student records with columns:
            - Name: Student name
            - Gender: 'Man' or 'Woman'
            - Knowledge: Numeric score
            - Skills: Numeric score
            - Attitude: Numeric score
            - Category: 'Superior' or 'Not Superior'

    LECTURE NOTE:
        pandas is Python's primary data analysis library. DataFrames are
        2D labeled data structures similar to spreadsheets or SQL tables.

        CSV (Comma-Separated Values) is a common format for storing
        tabular data. It's human-readable and widely supported.

        Best practices:
        - Always print confirmation after loading (verify success)
        - Check data dimensions (number of rows/columns)
        - Handle potential errors (file not found, corrupt data)
    ============================================================================
    """
    # Display loading message for user feedback
    print(f"Loading data from {file_path}...")

    # Read CSV file into pandas DataFrame
    data = pd.read_csv(file_path)

    # Confirm successful loading with record count
    print(f"Loaded {len(data)} student records.\n")

    return data


def preprocess_data(data):
    """
    ============================================================================
    DATA PREPROCESSING FUNCTION
    ============================================================================

    PURPOSE:
        Transform raw data into a format suitable for machine learning:
        - Encode categorical variables as numbers
        - Separate features (X) from target labels (y)
        - Convert to NumPy arrays for efficient computation

    PARAMETER:
        data (pandas.DataFrame): Raw student data from CSV

    RETURNS:
        tuple: (X, y) where:
            X (numpy.ndarray): Feature matrix, shape (n_students, 4)
                              [Gender_Encoded, Knowledge, Skills, Attitude]
            y (numpy.ndarray): Labels vector, shape (n_students,)
                              ['Superior', 'Not Superior', ...]

    FEATURE ENGINEERING:
        Gender Encoding:
        - 'Man' → 1
        - 'Woman' → 0
        - This binary encoding allows distance calculations
        - Alternative: One-hot encoding creates two columns

        Feature Selection:
        - Gender_Encoded: Demographic information
        - Knowledge: Academic understanding
        - Skills: Practical abilities
        - Attitude: Behavioral assessment

    LECTURE NOTE:
        Preprocessing is crucial in ML pipelines:
        1. Algorithms require numeric input (hence encoding)
        2. Feature scaling may be needed for some algorithms
           (KNN benefits from scaled features, though not applied here)
        3. Missing values must be handled (not present in this dataset)

        The .values attribute converts pandas Series to NumPy arrays,
        which are more efficient for mathematical operations.
    ============================================================================
    """
    # STEP 1: Encode categorical Gender variable
    # -------------------------------------------
    # Map string values to numeric codes for ML compatibility
    data['Gender_Encoded'] = data['Gender'].map({'Man': 1, 'Woman': 0})

    # STEP 2: Extract feature matrix (X)
    # -----------------------------------
    # Select 4 feature columns and convert to NumPy array
    # Shape: (n_students, 4) where each row is [Gender, Knowledge, Skills, Attitude]
    X = data[['Gender_Encoded', 'Knowledge', 'Skills', 'Attitude']].values

    # STEP 3: Extract target labels (y)
    # ----------------------------------
    # Get the classification labels we want to predict
    # Shape: (n_students,) - one label per student
    y = data['Category'].values

    return X, y


def display_data_summary(data):
    """
    ============================================================================
    EXPLORATORY DATA ANALYSIS (EDA) FUNCTION
    ============================================================================

    PURPOSE:
        Display comprehensive statistical summary of the dataset to
        understand data distribution, detect imbalances, and identify
        potential issues before modeling.

    PARAMETER:
        data (pandas.DataFrame): Student data to analyze

    OUTPUT:
        Prints formatted summary including:
        1. Class distribution (Superior vs Not Superior counts)
        2. Gender distribution (Man vs Woman counts)
        3. Score statistics (mean, std, min, max for each score type)

    WHY EDA MATTERS:
        - Reveals class imbalance (affects model performance)
        - Shows feature ranges (important for distance-based algorithms)
        - Identifies missing values or outliers
        - Helps understand what the model will learn from

    LECTURE NOTE:
        Always perform EDA before modeling! Common findings:
        - Imbalanced classes: May need SMOTE, class weights, or stratification
        - Different scales: May need normalization/standardization
        - Missing values: May need imputation or removal
        - Outliers: May need winsorization or removal

        For KNN specifically:
        - Feature scales matter! Large-scale features dominate distance
        - Imbalanced classes bias predictions toward majority class
        - This dataset uses scores on similar scales (good for KNN)
    ============================================================================
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
    ============================================================================
    MAIN EXECUTION FUNCTION: Complete ML Pipeline
    ============================================================================

    PURPOSE:
        Execute the complete machine learning workflow from data loading
        through model evaluation and results presentation.

    PIPELINE STAGES:
        1. Setup and initialization
        2. Data loading
        3. Exploratory data analysis
        4. Data preprocessing
        5. Train-test split
        6. Model training
        7. Prediction
        8. Evaluation
        9. Comparison with research paper
        10. Detailed results display

    LECTURE NOTE:
        This function demonstrates professional ML project structure:
        - Clear stage separation
        - User-friendly progress messages
        - Comprehensive logging
        - Results comparison with baseline
        - Detailed output for analysis

        In production systems, you might add:
        - Logging to files
        - Exception handling
        - Model serialization (saving/loading)
        - Hyperparameter tuning
        - Cross-validation
    ============================================================================
    """
    # STAGE 0: Setup and Configuration
    # =================================
    # Load configuration and prepare formatting variables
    cfg = CONFIG
    sep = "=" * cfg['separator_width']
    train_pct = int((1 - cfg['test_size']) * 100)
    test_pct = int(cfg['test_size'] * 100)

    # STAGE 1: Display Project Header
    # ================================
    print(f"\n{sep}")
    print(cfg['project_title'])
    print(cfg['school_name'])
    print(sep)

    # STAGE 2: Data Loading
    # ======================
    # Read student data from CSV file
    data = load_data(cfg['data_file'])

    # STAGE 3: Exploratory Data Analysis (EDA)
    # =========================================
    # Understand data distribution before modeling
    display_data_summary(data)

    # STAGE 4: Data Preprocessing
    # ============================
    # Transform raw data into ML-compatible format
    print("\nPreprocessing data...")
    X, y = preprocess_data(data)

    # STAGE 5: Train-Test Split
    # ==========================
    # Divide data: 95% training, 5% testing (stratified)
    print(f"Splitting data into train ({train_pct}%) and test ({test_pct}%) sets...")

    # Keep track of original indices to map back to student names later
    # This allows us to display which specific students were tested
    indices = np.arange(len(data))

    # Stratified split: maintains class distribution in both sets
    # If 60% Superior overall → ~60% Superior in train and test
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices,
        test_size=cfg['test_size'],
        random_state=cfg['random_state'],
        stratify=y  # Ensures balanced splits
    )

    # Display split sizes for verification
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # STAGE 6: Model Training
    # ========================
    # Initialize KNN classifier and "train" it (store training data)
    print(f"\n{sep}")
    print("TRAINING KNN CLASSIFIER")
    print(sep)
    print(f"Number of neighbors (k): {cfg['k_neighbors']}")

    # Create KNN instance with k=3 (as per research paper)
    knn = KNNClassifier(k=cfg['k_neighbors'])

    # Fit model to training data
    # For KNN, this simply stores the training examples in memory
    knn.fit(X_train, y_train)
    print("Training completed!")

    # STAGE 7: Prediction
    # ====================
    # Apply trained model to test data
    print(f"\n{sep}")
    print("MAKING PREDICTIONS")
    print(sep)

    # Generate predictions for all test samples
    # For each test student, KNN finds k nearest neighbors and votes
    y_pred = knn.predict(X_test)
    print(f"Predictions made for {len(y_test)} test samples.")

    # STAGE 8: Model Evaluation
    # ==========================
    # Calculate comprehensive performance metrics
    print(f"\n{sep}")
    print("MODEL EVALUATION")
    print(sep)

    # Compute confusion matrix and all metrics
    # This function displays results and returns a dictionary
    results = evaluate_model(y_test, y_pred, positive_label=cfg['positive_label'])

    # STAGE 9: Comparison with Published Results
    # ===========================================
    # Compare our implementation with the research paper's results
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

    # STAGE 10: Detailed Results Display
    # ===================================
    # Show individual predictions with student names
    print(f"\n{sep}")
    print(f"ALL {len(y_test)} TEST CASES WITH STUDENT NAMES")
    print(sep)

    # Table formatting
    name_width = cfg['name_column_width']
    cat_width = cfg['category_column_width']
    table_width = name_width + cat_width * 2 + 10

    # Table header
    print(f"{'Name':<{name_width}} {'Actual':<{cat_width}} {'Predicted':<{cat_width}} {'Correct':<10}")
    print("-" * table_width)

    # Display each test case with student name, actual label, prediction, and correctness
    for i in range(len(y_test)):
        # Map test index back to original DataFrame index
        original_idx = test_indices[i]
        # Get student name from original data
        name = data.iloc[original_idx]['Name']
        # Get actual and predicted labels
        actual = y_test[i]
        predicted = y_pred[i]
        # Mark correct (✓) or incorrect (✗)
        correct = cfg['correct_symbol'] if actual == predicted else cfg['incorrect_symbol']
        # Print formatted row
        print(f"{name:<{name_width}} {actual:<{cat_width}} {predicted:<{cat_width}} {correct:<10}")

    # Final completion message
    print(f"\n{sep}")
    print("Classification completed successfully!")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()

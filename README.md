# K-Nearest Neighbor Student Classification

Implementation of the research paper: **"Implementation of the K-Nearest Neighbor (kNN) Method to Determine Outstanding Student Classes"** by Nanda Fahrezi Munazhif, Gomal Juni Yanris, and Mila Nirmala Sari Hasibuan.

## Overview

This project implements a K-Nearest Neighbor (kNN) classifier to classify students into "Superior" and "Not Superior" categories based on their academic performance metrics.

### Research Context

- **Institution**: MTs Swasta Islamiyah Kotapinang
- **Dataset**: 60 student records
- **Features**: Knowledge scores, Skills scores, Attitude scores, Gender
- **Target**: Superior / Not Superior classification

## Project Structure

```
Labuhanbatu_KNN/
├── student_data.csv      # Dataset with 60 student records
├── knn_classifier.py     # KNN algorithm implementation
├── evaluation.py         # Evaluation metrics (Confusion Matrix, etc.)
├── main.py              # Main script to run classification
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

### 1. KNN Classifier (`knn_classifier.py`)
- Custom implementation of K-Nearest Neighbor algorithm
- Euclidean distance calculation
- Configurable k value (default: k=3)

### 2. Evaluation Metrics (`evaluation.py`)
- Confusion Matrix (TP, TN, FP, FN)
- Accuracy: `(TP + TN) / (TP + TN + FN + FP)`
- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1-Score

### 3. Data Processing
- CSV data loading
- Feature encoding (Gender: Man=1, Woman=0)
- Train-test split (80-20)
- Stratified sampling

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main classification script:

```bash
python main.py
```

This will:
1. Load the student data from `student_data.csv`
2. Display data summary and statistics
3. Train the KNN classifier (k=3)
4. Make predictions on test data
5. Display evaluation metrics
6. Compare results with the paper's findings

## Dataset

The dataset (`student_data.csv`) contains 60 student records with the following columns:

| Column     | Type    | Description                          |
|------------|---------|--------------------------------------|
| Name       | Text    | Student's full name                  |
| Gender     | Text    | Man or Woman                         |
| NIS        | Numeric | Student identification number        |
| Knowledge  | Numeric | Knowledge score (academic tests)     |
| Skills     | Numeric | Skills score (practical assessments) |
| Attitude   | Numeric | Attitude score (behavior)            |
| Category   | Text    | Superior or Not Superior (target)    |

## Expected Results

Based on the research paper, the expected performance metrics are:

- **Accuracy**: ~91.6%
- **Precision**: ~89.2%
- **Recall**: ~92.5%

Results may vary slightly due to:
- Random train-test split
- Different random state initialization
- Implementation variations

## Algorithm Details

### Euclidean Distance Formula

The KNN algorithm uses Euclidean distance to measure similarity between data points:

```
d(x,y) = √(Σ(x_i - y_i)²)
```

Where:
- `d(x,y)` = distance between points x and y
- `x_i`, `y_i` = individual feature values
- `Σ` = sum over all features

### Classification Process

1. Calculate distance from test point to all training points
2. Select k nearest neighbors (k=3)
3. Count class labels of k neighbors
4. Assign most common label to test point

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0 (used for train_test_split)

## References

Munazhif, N. F., Yanris, G. J., & Hasibuan, M. N. S. (2023). Implementation of the K-Nearest Neighbor (kNN) Method to Determine Outstanding Student Classes. *Sinkron: Jurnal dan Penelitian Teknik Informatika*, 8(2), 719-732. DOI: https://doi.org/10.33395/sinkron.v8i2.12227

## License

This project is for educational purposes based on the referenced research paper.

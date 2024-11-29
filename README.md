# ESA_PracticeProject

This project is focused on training and evaluating the performance of four classifiers: **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, **Decision Trees**, and **Logistic Regression (with SGD optimization)**. Hyperparameter optimization is applied to enhance the models, and their performance is compared before and after tuning.

### Table of Contents

- Overview
- Setup & Installation
- Usage

### Overview

The ESA_PracticeProject includes:
1. Data Preparation: Utilize pre-labeled datasets for image-based classification tasks.
2. Classifier Training: Implement four classifiers using scikit-learn:
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Trees
   - Logistic Regression (SGD-based)
3. Performance Evaluation: Measure accuracy and visualize results.
4. Hyperparameter Optimization: Use Grid Search (GridSearchCV) to tune each classifier for better performance.

### Setup & Installation

1. Clone the repository
2. Create a virtual environment: ```python3 -m venv venv```
3. Activate the virtual environment:
   - On macOS/Linux: ```source venv/bin/activate```
   - On Windows: ```.\venv\Scripts\activate```
4. Install required dependencies: ```pip install numpy pandas scikit-learn matplotlib```

### Usage

Individual classifier scripts can be executed to retrieve model specific results. Namely: 
- ```DecisionTree.py```
- ```KNN.py```
- ```LogisticRegression.py```
- ```SVM.py```
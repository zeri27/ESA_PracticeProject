import numpy as np
import pandas as pd
from common import labeled_images, labeled_digits, autograder_images
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Preprocess the Data
labeled_images_flat = labeled_images.reshape(3750, -1)
autograder_images_flat = autograder_images.reshape(len(autograder_images), -1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    labeled_images_flat, labeled_digits, test_size=0.2)
print('Training Data Samples: ', len(X_train))
print('Testing Data Samples: ', len(X_test))

# Train the SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Test the SVM Classifier (Accuracy)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Base SVM Accuracy: ', accuracy)

# Hyperparameter Optimization
param_grid = {'C': [0.1, 0.5, 1.0, 2.0],
              'gamma': [0.05, 0.1, 'scale', 'auto'],
              'kernel': ['linear', 'rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_params_)

# Train the Optimized SVM
svm_modelOP = grid.best_estimator_
# svm_modelOP = SVC(
#    kernel=grid.best_params_.get('kernel'),
#    C=grid.best_params_.get('C'),
#    gamma=grid.best_params_.get('gamma'))
# svm_modelOP.fit(X_train, y_train)

# Test the SVM-Optimized Classifier (Accuracy)
y_pred = svm_modelOP.predict(X_test)
accuracyOP = accuracy_score(y_test, y_pred)

print('Optimized SVM Accuracy: ', accuracyOP)
print('Difference between Default vs Optimized: ', accuracyOP - accuracy)
print('Negative means Default performed better')

# Use SVM Classifier on the Autograder dataset
prediction = svm_model.predict(autograder_images_flat)
result = np.append(accuracy, prediction)
pd.DataFrame(result).to_csv("autograder.txt", index=False, header=False)

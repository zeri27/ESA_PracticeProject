# ESA Group 28 : Project Report

## 1. Introduction

## 2. Data Exploration & Preprocessing
talk about initial data, its format, etc (28x28 matrix initially)
for each classifier, give some explanation.

For the SVM classifier, the dataset was transformed by flattening each 28x28 matrix into a 1D array of 784 values, creating a total of 3750 such arrays. This transformation was required to meet the SVM classifier's 2D array input format.

## 3. Regression with Default Hyperparameters

### SVM Classifier

The SVM classifier was trained using the default hyperparameters:
- ```C = 1.0``` Regularization parameter
- ```gamma = 'scale'``` Kernel co-efficient
- ```kernel = 'rbf'``` Radial Basis Function Kernel

The performance of the model was evaluated using accuracy on the test set and auto-grading via the scoring tool. The results were as follows:
- **Accuracy:** 95% on the test set.
- **Auto-grading score:** 100%.

The performance estimate of the model is fair because it was trained with default hyperparameters and evaluated on a separate 20% test subset of the dataset, ensuring it was tested on unseen data. Additionally, the model received a 100% score on the autograder, indicating strong generalization to new data.
## 4. Tuning with GridSearch

### SVM Classifier Optimization

The main hyperparameters chosen for optimization for SVM classifier are listed below:
- ```C``` (Regularization parameter): Low values of this parameter makes the decision boundary smoother, while higher values make it more complex: ```[0.1, 0.5, 1.0, 2.0]```
- ```gamma``` (Kernel co-efficient): Determines influence of each datapoint. Low values allow for less complex decision boundaries. The ```scale``` value scales the ```gamma``` value according to the training data. The ```auto``` value sets ```gamma``` based on the number of features: ```[0.05, 0.1, 'scale', 'auto']```
- ```kernel``` (Kernel): ```['linear', 'rbf']```

Once GridSearch is completed, the best hyperparameters are applied, and the model is evaluated on the test set to estimate its accuracy. The GridSearch resulted in a very small performance improvement of +0.01%. This minimal gain suggests that the model with the default hyperparameters already fits the data quite well.  
## 5. Conclusion
# ESA Group 28 : Project Report

## 1. Introduction

## 2. Data Exploration & Preprocessing
Data distribution among classes is roughly equal (smallest class has 366, while largest - 395 examples). As for pre-processing, we apply division by 255 (maximal channel value) and optionall apply scaler (where it's mentioned). Performance metric was accuracy among all conducted experiments.
Dummy estimator pre-trained using most frequent strategy scored 0.077 on test set serving as a baseline model.

![Example loaded data](/example_data.png)

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

### Logistic Regression Classifier
Standard scaler with default configuration was used for pre-processing data. 
Similarly to SVM, several C values (inverse of regularization strength) were explored, class weight was conditionally adjusted to respect slighly uneven classes distribution. 
Grid search also tried training models with or without bias parameter, adjust different iterations count and inspect multiple approaches how parameters are penalized (either L1 or L2 penalty were used).

The best discovered parameters setting:
* C: 0.1 `[out of 0.1, 0.5, 1.0, 2.0]`
* class_weight: balanced `[out of None, 'balanced']`
* fit_intercept: True `[out of True, False]`
* max_iter: 150 `[out of 150, 1000]`
* penalty: l2 `[out of 'l1', 'l2']`

- **Accuracy:** 90.4% on the test set.
- **Auto-grading score:** 30%.

### Decision Tree
Standard scaler with default configuration was used for pre-processing data. 
The grid search was employed to uptrain model using different impurity criterions (gini or entropy) and again, taking classes distributions into account when setting weight for classes.

Optimal parameters set found:
* class_weight: None `[out of None, 'balanced']`
* criterion: entropy `[out of 'gini', 'entropy']`

- **Accuracy:** 76.4% on the test set.
- **Auto-grading score:** 45%.

## 4. Tuning with GridSearch

### SVM Classifier Optimization

The main hyperparameters chosen for optimization for SVM classifier are listed below:
- ```C``` (Regularization parameter): Low values of this parameter makes the decision boundary smoother, while higher values make it more complex: ```[0.1, 0.5, 1.0, 2.0]```
- ```gamma``` (Kernel co-efficient): Determines influence of each datapoint. Low values allow for less complex decision boundaries. The ```scale``` value scales the ```gamma``` value according to the training data. The ```auto``` value sets ```gamma``` based on the number of features: ```[0.05, 0.1, 'scale', 'auto']```
- ```kernel``` (Kernel): ```['linear', 'rbf']```

Once GridSearch is completed, the best hyperparameters are applied, and the model is evaluated on the test set to estimate its accuracy. The GridSearch resulted in a very small performance improvement of +0.01%. This minimal gain suggests that the model with the default hyperparameters already fits the data quite well.  
## 5. Conclusion
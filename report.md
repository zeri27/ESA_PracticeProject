# ESA Group 28 : Project Report

## Part A. Data Exploration & Preprocessing

1. Data distribution among classes is roughly equal (smallest class has 366, while largest has 395 examples). 
   Performance metric was accuracy among all conducted experiments.

![Example loaded data](/example_data.png)

2. As for pre-processing, we apply division by 255 (maximal channel value) and optionally apply a scaler (where it's
   mentioned). For the SVM classifier, the dataset was transformed by flattening each 28x28 matrix into a 1D array of 784 values, creating a total of 3750 such arrays. This transformation was required to meet the SVM classifier's 2D array input format.

## Part B. Regression with Default Hyperparameters

3. Making a guess about the label of an image without knowing anything about the image would come down to assigning it
   some random number from 0 to 9 as a baseline.
   A dummy estimator pre-trained using the most-frequent strategy scored 0.077 (7.7%) on the test set.

### KNN Classifier

4. The default hyperparameters that were used for the KNN classifier are:

   - `k = 5` Number of neighbors
   - `weights = 'uniform'` Weighting function
   - `metric = 'minkowski'` Distance computation metric

5. Using the default hyperparameters, the following scores were obtained:

   - **Accuracy**: 91.33%.
   - **Auto-grading score**: 80/100.

### Logistic Regression Classifier

4. Standard scaler with default configuration was used for pre-processing data. 

5. [... scores with default settings]

### SVM Classifier

4. The SVM classifier was trained using the default hyperparameters:

   - ```C = 1.0``` Regularization parameter
   - ```gamma = 'scale'``` Kernel co-efficient
   - ```kernel = 'rbf'``` Radial Basis Function Kernel

5. The performance of the model was evaluated using accuracy on the test set and auto-grading via the scoring tool. The results were as follows:

   - **Accuracy:** 95% on the test set.
   - **Auto-grading score:** 100%.

   The performance estimate of the model is fair because it was trained with default hyperparameters and evaluated on a separate 20% test subset of the dataset, ensuring it was tested on unseen data. Additionally, the model received a 100% score on the autograder, indicating strong generalization to new data.

### Decision Tree

4. Standard scaler with default configuration was used for pre-processing data. 
   
5. [... scores with default settings]

## Part C. Tuning with GridSearch

### KNN Classifier

6. The hyperparameters that were considered in the grid search are:

   - `n_neighbors` with values `[1, 3, ..., 29]`. This resembles the `k` value. We used only odd values to prevent
     the risk of equal scores for multiple classes for single data points, and we stop at `k = 29` since the accuracy
     typically no longer increases once `k` reaches a certain value. As `k` increases, the decision boundaries typically
     become less complex.
   - `weights` with values `['uniform', 'distance']`. These values represent the weighting function to be used in the
     classification, where `'uniform'` assumes uniform weights for each data point, while `'distance'` assumes weights
     based on the inverse of the distance between data points.
   - `metric` with values `['minkowski', 'euclidean', 'manhattan', 'cosine']`, which determines how the distance values
     between data points are calculated.
   
   Applying grid search yields the hyperparameters that give the highest accuracy:
   `n_neighbors = 3`, `weights = 'distance'`, and `metric = 'cosine'`.
   The accuracy using these hyperparameters is 94.53%.

### Logistic Regression Classifier

6. Similarly to SVM, several `C` values (inverse of regularization strength) were explored, class weight was conditionally adjusted to respect slighly uneven classes distribution. 
   Grid search also tried training models with or without bias parameter, adjust different iterations count and inspect multiple approaches how parameters are penalized (either L1 or L2 penalty were used).

   The best discovered parameters setting:
   
   * `C`: 0.1 (out of `[0.1, 0.5, 1.0, 2.0]`)
   * `class_weight`: 'balanced' (out of `[None, 'balanced']`)
   * `fit_intercept`: True (out of `[True, False]`)
   * `max_iter`: 150 (out of `[150, 1000]`)
   * `penalty`: 'l2' (out of `['l1', 'l2']`)

   Scores using the optimized hyperparameters:
   
   - **Accuracy:** 90.4% on the test set.
   - **Auto-grading score:** 30%.

### SVM Classifier Optimization

6. The main hyperparameters chosen for optimization for SVM classifier are listed below:

   - ```C``` (Regularization parameter): Low values of this parameter makes the decision boundary smoother, while higher values make it more complex: ```[0.1, 0.5, 1.0, 2.0]```
   - ```gamma``` (Kernel co-efficient): Determines influence of each datapoint. Low values allow for less complex decision boundaries. The ```scale``` value scales the ```gamma``` value according to the training data. The ```auto``` value sets ```gamma``` based on the number of features: ```[0.05, 0.1, 'scale', 'auto']```
   - ```kernel``` (Kernel): ```['linear', 'rbf']```

   Once GridSearch is completed, the best hyperparameters are applied, and the model is evaluated on the test set to estimate its accuracy. The GridSearch resulted in a very small performance improvement of +0.01%. This minimal gain suggests that the model with the default hyperparameters already fits the data quite well.

### Decision Tree

6. The grid search was employed to uptrain model using different impurity criteria (gini or entropy) and again, taking classes distributions into account when setting weight for classes.

   Optimal parameters set found:
   * `class_weight`: None (out of `[None, 'balanced']`)
   * `criterion`: 'entropy' (out of `['gini', 'entropy']`)

   Scores using optimized hyperparameters:
   - **Accuracy:** 76.4% on the test set.
   - **Auto-grading score:** 45%.

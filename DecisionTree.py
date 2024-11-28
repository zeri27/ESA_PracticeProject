from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from common import labeled_images, labeled_digits, autograder_images
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

x = labeled_images.reshape(labeled_images.shape[0], -1)
x = x / 255.0
X_train, X_test, y_train, y_test = train_test_split(x, labeled_digits, random_state=0, test_size=0.2)
pipe = Pipeline([
    ('scaler', StandardScaler()),
                 ('decision',
                    DecisionTreeClassifier(min_impurity_decrease=0.0)
                  )
                 ])

param_grid = {
    'decision__criterion': ['gini', 'entropy'],
    'decision__class_weight': [None, 'balanced'],
}

grid = GridSearchCV(pipe, n_jobs=-1, param_grid=param_grid, cv=5,
                            scoring='accuracy', refit=True, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_params_)
print("grid.score(X_test, y_test)", grid.score(X_test, y_test))

be = grid.best_estimator_
y_pred = be.predict(X_test)
accuracyOP = accuracy_score(y_test, y_pred)
print("accuracyOP", accuracyOP)

autograder_images_flat = autograder_images.reshape(len(autograder_images), -1) / 255.0
prediction = be.predict(autograder_images_flat)
result = np.append(accuracyOP, prediction)
pd.DataFrame(result).to_csv("autograder.txt", index=False, header=False)

# from sklearn.dummy import DummyClassifier
# from sklearn.model_selection import cross_val_score
#
# clf = DummyClassifier(strategy='most_frequent', random_state=0)
# dummy_cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
# print(dummy_cv_scores)
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)
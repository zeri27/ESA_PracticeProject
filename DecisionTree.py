from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from common import labeled_images, labeled_digits, autograder_images
from sklearn.tree import DecisionTreeClassifier

x = labeled_images.reshape(labeled_images.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(x, labeled_digits, random_state=0)
pipe = Pipeline([('scaler', StandardScaler()),
                 ('decision',
                    DecisionTreeClassifier()
                  )
                 ])

param_grid = {
    # 'decision__fit_intercept': [True, False],
}

grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv=5,
                            scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
grid.score(X_test, y_test)

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

clf = DummyClassifier(strategy='most_frequent', random_state=0)
dummy_cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print(dummy_cv_scores)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
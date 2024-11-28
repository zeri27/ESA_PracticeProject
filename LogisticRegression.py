from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from common import labeled_images, labeled_digits, autograder_images
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

x = labeled_images.reshape(labeled_images.shape[0], -1)
x = x / 255.0
X_train, X_test, y_train, y_test = train_test_split(x, labeled_digits, random_state=0, test_size=0.2)
pipe = Pipeline([
                ('scaler', StandardScaler()),
                 ('logistic',
                  LogisticRegression(max_iter = 150, tol=1e-3)
                    # SGDClassifier(loss="log_loss", max_iter=150, tol=1e-3)
                  )
                 ])
#pipe.fit(X_train, y_train).score(X_test, y_test)

param_grid = {
    'logistic__fit_intercept': [True, False],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__C': [0.1, 0.5, 1.0, 2.0],
    'logistic__class_weight': [None, 'balanced'],
    'logistic__max_iter': [150, 1000],
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

# import pandas as pd
#
# mean_scores = np.array(grid.cv_results_["mean_test_score"])
# mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# mean_scores = mean_scores.max(axis=0)
# mean_scores = pd.DataFrame(
#     mean_scores.T, index=N_FEATURES_OPTIONS, columns=reducer_labels
# )
# ax = mean_scores.plot.bar()
# ax.set_title("Comparing feature reduction techniques")
# ax.set_xlabel("Reduced number of features")
# ax.set_ylabel("Digit classification accuracy")
# ax.set_ylim((0, 1))
# ax.legend(loc="upper left")
# plt.show()

# from sklearn.dummy import DummyClassifier
# from sklearn.model_selection import cross_val_score
#
# clf = DummyClassifier(strategy='most_frequent', random_state=0)
# dummy_cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
# print(dummy_cv_scores)
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)
#
# from sklearn import preprocessing
# import numpy as np
# X_train = np.array([[ 1., -1.,  2.],
#                     [ 2.,  0.,  0.],
#                     [ 0.,  1., -1.]])
# scaler = preprocessing.StandardScaler().fit(X_train)
# # scaler
# # scaler.mean_
# # scaler.scale_
# X_scaled = scaler.transform(X_train)
# X_scaled

# y_pred = clf.predict(x)
# accuracy = np.mean(y_pred == labeled_digits)
# print('Accuracy on the training set:', accuracy)
# y_pred = clf.predict(X_test)
# accuracy = np.mean(y_pred == y_test)
# print('Accuracy on the test set:', accuracy)
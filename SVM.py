from common import labeled_images, labeled_digits, autograder_images
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Preprocess the Data
labeled_images_flat = labeled_images.reshape(3750, -1)
labeled_images_flat = labeled_images_flat / 255.0
autograder_images_flat = autograder_images.reshape(len(autograder_images), -1) / 255.0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    labeled_images_flat, labeled_digits, test_size=0.2)
print(len(X_train))
print(len(X_test))

### Train the SVM classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Test the SVM Classifier (Accuracy)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Use SVM Classifier on the Autograder dataset
# prediction = svm_model.predict(autograder_images_flat)
# result = np.append(accuracy, prediction)
# pd.DataFrame(result).to_csv("autograder.txt", index=False, header=False)

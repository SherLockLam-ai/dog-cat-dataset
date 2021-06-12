import numpy as np 
import os 
import cv2 
import glob2 

from sklearn.metrics import accuracy_score 
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def image_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()

def predict_image(image_path):
    image = cv2.imread(image_path) 
    image = image_vector(image) 
    image = image.reshape(1, -1) 
    predict = LR_model.predict(image) 
    if predict[0] == 0:
        return 'cat'
    else:
        return 'dog'

train_cats_path = "data/train/cats" 
train_dogs_path = "data/train/dogs"
test_cats_path = "data/test/cats" 
test_dogs_path = "data/test/dogs"

list_image_train_cats = glob2.glob(os.path.join(train_cats_path, "*.jpg"))
list_image_train_dogs = glob2.glob(os.path.join(train_dogs_path, "*.jpg"))
list_image_test_cats = glob2.glob(os.path.join(test_cats_path, "*.jpg"))
list_image_test_dogs = glob2.glob(os.path.join(test_dogs_path, "*.jpg"))

list_image_train_cats = [cv2.imread(image_path) for image_path in list_image_train_cats]
list_image_train_dogs = [cv2.imread(image_path) for image_path in list_image_train_dogs]
list_image_test_cats = [cv2.imread(image_path) for image_path in list_image_test_cats]
list_image_test_dogs = [cv2.imread(image_path) for image_path in list_image_test_dogs]

list_image_train_cats = [image_vector(image) for image in list_image_train_cats]
list_image_train_dogs = [image_vector(image) for image in list_image_train_dogs]
list_image_test_cats = [image_vector(image) for image in list_image_test_cats]
list_image_test_dogs = [image_vector(image) for image in list_image_test_dogs]

list_image_train_cats = np.array(list_image_train_cats)
list_image_train_dogs = np.array(list_image_train_cats)
list_image_test_cats = np.array(list_image_test_cats)
list_image_test_dogs = np.array(list_image_test_dogs)

list_labels_train_cats = np.zeros(len(list_image_train_cats), dtype=int)
list_labels_train_dogs = np.ones(len(list_image_train_dogs), dtype=int)
list_labels_test_cats = np.zeros(len(list_image_test_cats), dtype=int)
list_labels_test_dogs = np.ones(len(list_image_test_dogs), dtype=int)

X_train = np.concatenate((list_image_train_cats, list_image_train_dogs), axis=0)
y_train = np.concatenate((list_labels_train_cats, list_labels_train_dogs), axis=0)
X_test = np.concatenate((list_image_test_cats, list_image_test_dogs), axis=0)
y_test = np.concatenate((list_labels_test_cats, list_labels_test_dogs), axis=0)

# Perceptron
PLA_model = Perceptron(tol=1e-3) 
PLA_model = PLA_model.fit(X_train, y_train) 
y_predict = PLA_model.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)
print("ACC: ", accuracy)

# KNN
KNN_model = KNeighborsClassifier(n_neighbors=5) 
KNN_model.fit(X_train, y_train)
y_predict = KNN_model.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)
print("ACC: ", accuracy)

# Logistic Regression
LR_model = LogisticRegression() 
LR_model.fit(X_train, y_train)
y_predict = LR_model.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)
print("ACC: ", accuracy)


# SVM
SVC = LinearSVC()
SVC.fit(X_train, y_train)
y_predict = SVC.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)
print("ACC: ", accuracy)

# Random Forest
RF_model = RandomForestClassifier(max_depth=2, random_state=0)
RF_modelfit(X_train, y_train)
y_predict = RF_modelfit.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)
print("ACC: ", accuracy)
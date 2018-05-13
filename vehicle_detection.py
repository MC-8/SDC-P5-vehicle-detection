# Main file for the project
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utilities import scale_features, draw_boxes, color_hist, bin_spatial, get_hog_features, extract_features

plt.interactive(False)
# Prepare data, first read images of cars and not cars, then extract features,
# the labels will just be a 1 for CAR and a 0 for "NOT CAR"

# Read images from dataset
img_cars_files = glob.glob("dataset\\vehicles_smallset\\**\\*.jpeg",recursive= True)
print(img_cars_files[0:6])
img_notcars_files = glob.glob("dataset\\non-vehicles_smallset\\**\\*.jpeg",recursive= True)
print(img_notcars_files[0:6])

img_cars = []
img_not_cars = []

for image_file in img_cars_files[0:6]:
    img_cars.append(image_file)
for image_file in img_notcars_files[0:6]:
    img_not_cars.append(image_file)

# Extract features from images of cars and not-cars
car_features = extract_features(img_cars)
notcar_features = extract_features(img_not_cars)

print(np.shape(car_features))
print(np.shape(notcar_features))

X1 = np.vstack(car_features)
X2 = np.vstack(notcar_features)
X = np.vstack((X1, X2))

# Create corresponding labels
y = np.hstack((np.ones(len(car_features)),
              np.zeros(len(notcar_features))))

# Create an array stack of feature vectors
print(np.shape(X))
## Split dataset into training and test set
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

print(np.shape(X_train))
print(np.shape(X_test))

# Scale features
X_scaler = StandardScaler().fit((X_train))
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

scaled_X_train = scale_features(X_train)
scaled_X_test = scale_features(X_test)


print(np.shape(scaled_X_train))
print(np.shape(scaled_X_test))

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svr = svm.SVC()
#clf = model_selection.GridSearchCV(svr, parameters)
#clf.fit(iris.data, iris.target)
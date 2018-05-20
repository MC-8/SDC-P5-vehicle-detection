# Main file for the project
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# Now to detect images, get the frame, iterate (windows) then check if there is a car in the box!
# test car detection on test images
from utilities import slide_window, draw_boxes
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Train network
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from utilities import draw_boxes, extract_features, add_heat, apply_threshold, draw_labeled_bboxes
from time import time
import PIL

plt.interactive(False)
# Prepare data, first read images of cars and not cars, then extract features,
# the labels will just be a 1 for CAR and a 0 for "NOT CAR"

# Read images from dataset
# Small dataset
#img_cars_files = glob.glob("dataset\\vehicles_smallset\\**\\*.jpeg",recursive= True)
#img_notcars_files = glob.glob("dataset\\non-vehicles_smallset\\**\\*.jpeg",recursive= True)
# Large dataset
img_cars_files = glob.glob("dataset\\vehicles\\**\\*.png",recursive= True)
img_notcars_files = glob.glob("dataset\\non-vehicles\\**\\*.png",recursive= True)

img_cars = []
img_not_cars = []
for image_file in img_cars_files:
    img_cars.append(image_file)
for image_file in img_notcars_files:
    img_not_cars.append(image_file)

# Extract features from images of cars and not-cars
car_features = extract_features(img_cars)
notcar_features = extract_features(img_not_cars)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Create corresponding labels
y = np.hstack((np.ones(len(car_features)),
              np.zeros(len(notcar_features))))
n_classes = 2

## Split dataset into training and test set
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Scale features
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X_train and X_test
scaled_X_train = X_scaler.transform(X_train)
scaled_X_test = X_scaler.transform(X_test)

########## Use GridSearchCV to find good parameters for SV classifier #############
# Check the training time for the SVC
# GridSeachCV Is very slow, but I can use it to find the best parameters for SVC
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100], }
#parameters = [
#  {'C': [1, 10, 100], 'kernel': ['linear']},
#  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 0.00001], 'degree': [2, 3, 4], 'kernel': ['rbf']},
# ]
#clf = GridSearchCV(svc, parameters, return_train_score=False, n_jobs=8)
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)
###################################################################################

# Classifier with parameter set found via GridSearchCV
clf = svm.SVC(C=10, cache_size=1000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=2, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.005, verbose=False)

print("Fitting the classifier to the training set")
t0 = time()
clf.fit(scaled_X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting...")
t0 = time()
y_pred = clf.predict(scaled_X_test)
print("done in %0.3fs" % (time() - t0))

print("Classification report")
print(classification_report(y_test, y_pred, target_names=['Car','Not Car']))

print("Confusion matrix")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

def detect_cars(frame):
    detected_windows = []
    # Classification step
    X_show = True
    for window in windows:
        cropped_fig = frame[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        # need to resize to a 64 by 64 image to classify
        dst = cv2.resize(cropped_fig, (64, 64), interpolation=cv2.INTER_NEAREST)
        # Classify
        X = extract_features(dst,
                             c_space='HLS',
                             spatial_size=(32, 32),
                             hist_bins=32,
                             hist_range=(0, 256),
                             hog_channels=2,
                             hog_orient=8,
                             hog_pix_per_cell=8,
                             hog_cell_per_block=2)
        # Apply the scaler to X_train and X_test
        scaled_X = X_scaler.transform([np.array(X)])
        Y_pred = clf.predict(scaled_X)
        if X_show:
            #print("Mean Std X: {}, {}".format(np.mean(X), np.std(X)))
            #print("Scald MS X: {}, {}".format(np.mean(scaled_X), np.std(scaled_X)))
            X_show = False
        if Y_pred:
            detected_windows.append(window)
            # cv2.rectangle(frame, (window[0][0],window[1][0]),(window[0][1],window[1][1]), (0, 0, 255), 6)

    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heatmap = add_heat(heat, detected_windows)

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, threshold=1)

    # Clip heatmap
    heatmap = np.clip(heatmap, 0, 255)
    #print(np.amin(heatmap))
    #print(np.amax(heatmap))
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(frame), labels)

    heatmap *= 255
    heatmap = np.uint8(np.clip(heatmap, 0, 255))

    # Add heatmap to frame (top left corner)
    img_w, img_h = (320, 240)
    bg_w, bg_h = (1280, 720)
    offset = ((bg_w - img_w) // 8, (bg_h - img_h) // 8)
    heatmap_small = cv2.resize(heatmap, (320, 240))
    heatmap_small_pip = PIL.Image.fromarray(heatmap_small)

    draw_img_pip = PIL.Image.fromarray(draw_img)
    draw_img_pip.paste(heatmap_small_pip, offset)
    draw_img_return = np.array(draw_img_pip)
    # draw_img = draw_boxes(frame, detected_windows, thickness=3)
    return draw_img_return

# Define windows where to search

test_image = mpimg.imread("test_images\\test2.jpg")
img = np.zeros_like(test_image,dtype= np.uint8)
windows = slide_window(img,
                       (int(test_image.shape[1]*1/3), test_image.shape[1]),
                       (int(test_image.shape[0] / 2), int(test_image.shape[0]*8/10)),
                       xy_window=(int(test_image.shape[0] / 4), int(test_image.shape[0] / 4)))

for den in [6, 8, 10]:
    windows += slide_window(test_image,
                            (int(test_image.shape[1]*1/3), test_image.shape[1]),
                            (int(test_image.shape[0] / 2), int(test_image.shape[0]*8/10)),
                            xy_window=(int(test_image.shape[0] / den), int(test_image.shape[0] / den)))

# Explore data
imcopy = draw_boxes(test_image, windows, thickness=3)


# Test algorithm on test images

test_images = glob.glob('./test_images/test*.jpg')
fig, axs = plt.subplots(3, 2, figsize=(16,9))
fig.subplots_adjust(hspace = .004, wspace=.004)
axs = axs.ravel()

for i, im in enumerate(test_images[0:6]):
    axs[i].imshow(detect_cars(mpimg.imread(im)))
    axs[i].axis('off')
plt.show()

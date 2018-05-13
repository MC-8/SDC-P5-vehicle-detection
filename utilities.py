# Utility functions for vehicle detection project
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def draw_boxes(img, b_boxes, color=(0, 0, 255), thickness=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for b_box in b_boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, b_box[0], b_box[1], color, thickness)
    # Return the image copy with boxes drawn
    return draw_img


# Define a function to compute color histogram features
def color_hist(img, n_bins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=n_bins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=n_bins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=n_bins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    #return rhist, ghist, bhist, bin_centers, hist_features
    return hist_features


# Define a function that takes an image, a color space,
# and a new image size
# and returns a feature vector
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False,
                     feature_vec=True):
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """

    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys', transform_sqrt=False,
                      visualise=vis, feature_vector=feature_vec)

    # name returns explicitly
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features


# Define a function to extract features from a list of images
def extract_features(imgs, c_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create empty list to append features to
    features = []
    # Iterate through the list of images
    print("Extracting features...")
    for file in tqdm(imgs):
        # Read in each one by one
        img = mpimg.imread(file)    # Make a copy of the current image argument
        feature_image = np.copy(img)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, color_space=c_space, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, n_bins=hist_bins, bins_range=hist_range)
        # Apply hog()
        hog_features = get_hog_features(cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY))
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, [hog_features])))
        #features.append(hog_features)#, hist_features, hog_features)))
        #features.append(np.concatenate((spatial_features, hist_features)))
        # Return list of feature vectors
    return features


# Function to scale a vector of features (so that all count equally)
def scale_features(features):
    # Fit a per-column scaler
    feature_scaler = StandardScaler().fit(features)
    # Apply the scaler to X
    scaled_features = feature_scaler.transform(features)
    return scaled_features
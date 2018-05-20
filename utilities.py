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
def bin_spatial(img, color_space='YUV', size=(32, 32)):
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


def get_hog_features(img, orient=8, pix_per_cell=8, cell_per_block=2, vis=False,
                     feature_vec=True, color_space='YUV', hog_channels=1):
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """
    hog_features = []
    hog_image = []
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
    if (hog_channels == 'ALL'):
        channels = [0,1,2]
    else:
        channels = [int(hog_channels)]

    for channel in channels:
        return_list = hog(feature_image[:,:,channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block),
                          block_norm='L2-Hys', transform_sqrt=False,
                          visualise=vis, feature_vector=feature_vec)

        # name returns explicitly
        hog_features.append(return_list[0])
        if vis:
            hog_image.append(return_list[1])
            
    if vis:
        return hog_features, hog_image
    else:
        return hog_features


# Define a function to extract features from a list of images
def extract_features(imgs,
                    c_space='YUV',
                    spatial_size=(32, 32),
                    hist_bins=32,
                    hist_range=(0, 256),
                    hog_channels=2,
                    hog_orient=11,
                    hog_pix_per_cell=16,
                    hog_cell_per_block=2):

    # Create empty list to append features to
    features = []

    if (isinstance(imgs, np.ndarray)): # Single image
        feature_image = np.copy(imgs)
        feature_image = feature_image.astype(np.float32) / 255
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, color_space=c_space, size=spatial_size)

        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, n_bins=hist_bins, bins_range=hist_range)

        # Apply hog()
        hog_features = get_hog_features(feature_image,
                                        orient=hog_orient,
                                        pix_per_cell=hog_pix_per_cell,
                                        cell_per_block=hog_cell_per_block,
                                        color_space=c_space,
                                        hog_channels=hog_channels)

        # Append the new feature vector to the features list
        if hog_channels != 'ALL':
            features = np.concatenate((spatial_features, hist_features, hog_features))
        else:
            features =  np.concatenate((spatial_features, hist_features, hog_features))

    elif (isinstance(imgs, list)):
        for file in tqdm(imgs):
            # Read in each one by one
            img = mpimg.imread(file)    # Make a copy of the current image argument
            feature_image = np.copy(img)
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, color_space=c_space, size=spatial_size)

            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, n_bins=hist_bins, bins_range=hist_range)

            # Apply hog()
            hog_features = get_hog_features(feature_image,
                                            orient=hog_orient,
                                            pix_per_cell=hog_pix_per_cell,
                                            cell_per_block=hog_cell_per_block,
                                            color_space=c_space,
                                            hog_channels=hog_channels)

            # Append the new feature vector to the features list
            if hog_channels != 'ALL':
                features.append(np.concatenate((spatial_features, hist_features, hog_features)))
            else:
                features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    else:
        assert(False)
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=(0, 0), y_start_stop=(0, 0),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[1] == 0:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[1] == 0:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


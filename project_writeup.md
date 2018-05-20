## Project Writeup


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[cars]:   ./writeup_images/cars.png
[colorspaces]: ./writeup_images/Color-spaces.png
[windows]: ./writeup_images/windows.png
[processedtestimages]: ./writeup_images/processed_test_images.png
[noncars]: ./writeup_images/non-cars.png
[hog]: ./writeup_images/HOG.png
[video]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function in the `utilities.py` file. The feature extraction (in the `extract_features` function), however, also extracts color spatial and histogram information.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of  `vehicle` class:
![alt text][cars]

and `non-vehicle` class:
![alt text][noncars]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=8` and `cells_per_block=2`:


![alt text][HOG]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameter, that is varying the colorspace and selecting different colorspace channels, and trying different values for orientations and pixels per cell. First I tried to compare the accuracy score of a linear classifier, trained from features coming from a small dataset, and several combination of HOG parameter values. 
What I obtaines was that with most of the training I would obtain more than 99% accuracy, although the final video output had many false positives. 
I noticed that the small dataset did not have in the non-car class many of the characteristics typical of the project video, that is, yellow lines and changes in asphalt color and contrast.
Tuning the HOG parameters and selecting the right colorspace is what made the difference.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before selecting a SVM I used `GridSeachCV` to find a set of parameters that would yield a great accuracy for a classifier. Here are the set of parameters that I tried:
```python
parameters = [
  {'C': [1, 10, 100], 'kernel': ['linear']},
  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 0.00001],  'degree': [2, 3, 4], 'kernel': ['rbf']},
]
```
And this is the classifier that yielded the best results:

```python
clf = svm.SVC(C=10, 
              degree=2, 
              gamma=0.0001,
              kernel='rbf',
              tol=0.005, 
              verbose=False)
```

This is in the 2nd code cell in the `vehicle_detection.ipynb` notebook.
The feature extraction functions are in the `utilities.py` file, whether the main file with the detection logic is in the `vehicle_detection.ipynb` notebook.

The car detection pipeline is in the 3rd code cell in in the `vehicle_detection.ipynb` notebook. 
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the center area of the image using multiple window sizes, that are a fraction of the screen. I used the y-axis size `720` as a starting point and divided that by `4`, `6`, `8`, `10` to obtain 4 sets of windows of sizes `180x180`, `120x120`, `90x90`, and `72x72`. This is in the 4th code cell in the `vehicle_detection.ipynb` notebook.

I also chose to restrict the search location by not looking for cars on the left side to avoid many false positives or detecting incoming cars on the left side. In a scenario where the car changes lane and needs to keep an eye on all sides of the road, clearly this strategy would be catastrophic.
Each window is then resized to `64x64` pixels using openCV, which is the same size of the training set images. 
For each resized images I then extract features and classify them to identify if the window contains a car or not.
![alt text][windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3rd channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][processedtestimages]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

 From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. I set the threshold to `1` that is sporadic and isolated detections (no window overlap) would be ignored.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
The heatmap is represented in each frame as a black and white control box on the top left corner.

![alt text][processedtestimages]

The control box contains integrated heatmaps, and the frames show the bounding boxes that are generated by taking the extreme coordinates of each "blob".

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is extremely slow, it takes more than 1 hour to process the video on a good CPU (i7-7700hq).
One way to optimize it is to calculate the hog features of the whole searched region, rather calculating hog features at each window. Also the pipeline still present some false positives and the car detection stops working when cars are a bit far.
The first issue could be resolved by delaying the detection by one or more frame, once there is some persistence on the heatmap. The second issue can be resolved by using smaller window, at the cost of increasing the computational requirements.
Thus, to improve the pipeline performance one way would be not to search each window at each frame for cars, instead, if there are cars at a specific frame and at a specific location, the algorithm should take into account of the current position and try to look for cars in the next frame in the vicinity of the position where a car was previously detected.
Similarly, cars don't suddenly "pop-up" and disappear between frames. The pipeline may have "candidate" detected cars which, if persisting for enough frames are then confermed as cars. This would help having persistence in the detection, and eliminate false positives. Systematic false positives due to asphalt conditions should be tackled by improving the computer vision algorithm instead.


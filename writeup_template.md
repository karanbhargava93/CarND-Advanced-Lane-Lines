**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[camera_calib]: ./output_images/camera_calib.jpeg "Undistorted"
[original]: ./output_images/original.jpeg "Original"
[undist_img]: ./output_images/undistorted.jpeg "Undistorted Image in Pipeline"
[threshold]: ./output_images/threshold.jpeg "Threshold the image to find lane lines"
[warped]: ./output_images/warped.jpeg "Birds Eye View"
[result]: ./output_images/result.jpeg "Output"
[poly]: ./output_images/poly.jpeg "Curve fitting"
[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the result below. 

To summarize, the code for the calibration is given in the camera_cal/cal_script , it saves the parameters into a pickle file for accessing it later. OpenCV functions were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository. The distortion matrix was used to un-distort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is given below.

![alt text][camera_calib]

### Pipeline (single images)

#### 1. Undistort the image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][undist_img]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #8 through #20 in `functions.py` and also #128 through #136 for converting it into the HLS colorspace).  Here's an example of my output for this step.

![alt text][threshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src= np.float32([[540,  400],
                 [210,  580],
                 [1050, 580],
                 [690,  400]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 400      | 303, 0        | 
| 210, 580      | 303, 628      |
| 1050, 580     | 909, 628      |
| 690, 400      | 909, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is an example of the case where the road curves away.

![alt text][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the './code/functions.py', lines #185 through #242 in the 'process_image()' function identify the location of the centroids of the window. After this I fitted the (x,y) coordinates of the left and right lanes. The identified lanes and the fitted second order polynomial is shown in the image below.

![alt text][poly]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did the curvature calculation in lines #258 through #274 in my code in `./code/functions.py`. And the position of the center is calculated in lines #298 through #300 in './code/functions.py'.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #113 through #376 in my code in `./code/functions.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major problem in this particular pipeline is that its very dependent on how we set the thresholds. This tuning took a fair bit of time. Moreover, these tuned parameters will have to be changed depending on the video frame. So some sort of adaptive tuning of the thresholds is needed. This pipeline doesn't perform upto the mark on the challenge video majorly because of its inability to adjust threholds wherever needed. Another solution for the same would be to train a neural network which would process the image for the lane lines and give us the curvature and the deviation from the center of the lane.


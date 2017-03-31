import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

from functions import *

with open('calibparams.pickle', 'rb') as handle:
    K = pickle.load(handle)
#del K
mtx = K['mtx']
dist = K['dist']
newcameramtx = K['newcameramtx']
roi = K['roi']

# print(mtx)
# print(dist)

# Original image
img = mpimg.imread('../test_images/test5.jpg') # test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # 

# Undistort image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]

undistorted_img = dst

ksize = 3

hls = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

gray = S

gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 100))#(50, 255))
grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 100))#(100, 255))
mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 255))#(70, 255))
dir_binary = dir_thresh(gray,  sobel_kernel=ksize, thresh=(30, 60)) #(45, 57, 120, 140)) #np.pi/2))

combined = gradx
combined = np.zeros_like(dir_binary)

thresholdRmin = 150
thresholdRmax = 255
R = S #undistorted_img[:,:,0]
R_binary = np.zeros_like(R)
R_binary[(R > thresholdRmin) & (R < thresholdRmax)] = 1

combined[((gradx == 1) | (grady == 1)) & ((mag_binary == 1) | (dir_binary == 1)) & (R_binary == 1)] = 1

plt.figure()
plt.imshow(undistorted_img)

plt.figure()
plt.imshow(combined, cmap = 'gray')
plt.title('R_binary')

# plt.figure()
# plt.imshow(combined, cmap = 'gray')
# plt.title('combined')



plt.show()
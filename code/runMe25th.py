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
img = mpimg.imread('../test_images/test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # test5.jpg') # 

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



# Perspective transform

img_size = gray.shape[::-1]

src = np.float32(   [[566.0, 394.0],
                     [212.0, 598.0],
                     [1050.0, 598.0],
                     [666.0, 394.0]])
dst = np.float32(   [[(img_size[0] / 4), 0],
                    [(img_size[0] / 4), img_size[1]],
                    [(img_size[0] * 3 / 4), img_size[1]],
                    [(img_size[0] * 3 / 4), 0]])

#print(src)
M = cv2.getPerspectiveTransform(src, dst)
print('M =', M)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2) #, figsize=(24, 9))
f.tight_layout()
ax1.imshow(undistorted_img)
ax1.plot(src[0,0], src[0,1], '.')
ax1.plot(src[1,0], src[1,1], '.')
ax1.plot(src[2,0], src[2,1], '.')
ax1.plot(src[3,0], src[3,1], '.')
ax1.set_title('Original Image') #, fontsize=50)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Thresholded Gradient') #, fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# plt.figure()
# plt.imshow(warped, cmap = 'gray')



# Fit polynomials

# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 50 # How much to slide left and right for searching

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows    
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channle 
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)


y = np.linspace(int(window_height/2), gray.shape[0]-int(window_height/2), num = 7)
y = y[::-1]
window_centroids = np.array(window_centroids)
xl = np.array(window_centroids[:,0])
xr = np.array(window_centroids[:,1])

print('y shape =', y.shape)
print('x shape =', xl.shape)


# Display the final results
# warpedimg = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
# plt.imshow(warped, cmap='gray')
# #plt.plot(xl, y, '.', markersize = 50)
# plt.title('window fitting results')

#plt.show()

# Fit a second order polynomial to pixel positions in each fake lane line

left_fit = np.polyfit(y, xl, 2)
left_fitx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
right_fit = np.polyfit(y, xr, 2)
right_fitx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

# # Plot up the fake data


mark_size = 30
plt.imshow(warped, cmap = 'gray')
plt.plot(xl, y, 'o', color='red', markersize=mark_size)
plt.plot(xr, y, 'o', color='blue', markersize=mark_size)


# plt.xlim(0, 1280)
# plt.ylim(0, 720)
# plt.plot(left_fitx, y, color='yellow', linewidth=3)
# plt.plot(right_fitx, y, color='yellow', linewidth=3)
# plt.gca().invert_yaxis() # to visualize as we do the images


# Curvature estimation

y_eval = gray.shape[0] #np.max(y)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 3.0/72 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(y*ym_per_pix, xl*xm_per_pix, 2)
right_fit_cr = np.polyfit(y*ym_per_pix, xr*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
curve = np.array([left_curverad, right_curverad])
# Now our radius of curvature is in meters
curvature = np.min(curve)
print('Radius of Curvature =', curvature, 'm')



# Overlay
midx = 650

ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0])) 


# Combine the result with the original image
result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
xroi,yroi,wroi,hroi = roi
result = result[yroi:yroi+hroi, xroi:xroi+wroi]

cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
if position_from_center < 0:
    text = 'left'
else:
    text = 'right'
cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

plt.figure()
plt.imshow(result)
plt.title('result')

# plt.figure()
# plt.imshow(warped, cmap = 'gray')
# plt.title('warped')

plt.show()
cv2.destroyAllWindows()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):

        # Is this the first frame?
        self.first_frame = True
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        self.points_left_avg  = [np.array([False])]
        self.points_right_avg = [np.array([False])]
        #polynomial coefficients for the most recent fit
        self.left_fit_cr = [np.array([False])]  
        self.right_fit_cr = [np.array([False])] 
        #radius of curvature of the line in some units
        self.curvature = None 
        #distance in meters of vehicle center from the line
        self.offset = None 


        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, left_fit_cr, right_fit_cr, curvature, points_left_avg, points_right_avg):
        if (self.first_frame == True):
            
            self.first_frame = False
            # self.detected = True

            # self.left_fit_cr = left_fit_cr  
            # self.right_fit_cr = right_fit_cr
            self.curvature = curvature

            self.points_left_avg  = points_left_avg
            self.points_right_avg = points_right_avg

        else:
            thresh = 10
            theta = 0.25
            if ((abs(self.points_left_avg - points_left_avg) <= thresh) & (abs(self.points_right_avg - points_right_avg) <= thresh)):
                # Lanes detected
                self.detected = True
                self.points_left_avg = (1-theta)*self.points_left_avg + theta*points_left_avg
                self.points_right_avg = (1-theta)*self.points_right_avg + theta*points_right_avg
                self.curvature = (1-theta)*self.curvature + theta*curvature
            else:
                self.detected = False
                # Don't update values








def abs_sobel_thresh(gray, orient, sobel_kernel, thresh):
    
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    mag = np.uint8(255*mag/np.max(mag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(mag)
    binary_output[(mag>mag_thresh[0]) & (mag<mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    thresh = np.radians(thresh)
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    theta = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gray)
    binary_output[((theta > thresh[0]) & (theta < thresh[1]))] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    desired_frac = 1/3
    frac = 1 - desired_frac
    l_sum = np.sum(warped[int(frac*warped.shape[0]):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(frac*warped.shape[0]):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))

        if (np.max(conv_signal[l_min_index:l_max_index]) > 0):
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        else:
            l_center = int((l_max_index + l_min_index)/2) - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))

        if (np.max(conv_signal[r_min_index:r_max_index]) > 0):
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        else:
            r_center = int((r_max_index+r_min_index)/2)-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def process_image(img):

    # line = Line()

    with open('calibparams.pickle', 'rb') as handle:
        K = pickle.load(handle)
    mtx = K['mtx']
    dist = K['dist']
    newcameramtx = K['newcameramtx']
    roi = K['roi']

    # Undistort image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    undistorted_img = dst

    # Threshold image

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

    combined = gradx #mag_binary
    combined = np.zeros_like(dir_binary)
    combined[((mag_binary == 1) & (dir_binary == 1))] = 1  #[((gradx == 1) | (grady == 1)) & ((mag_binary == 1) | (dir_binary == 1))] = 1

    # # Plot the result
    # f, (ax1) = plt.subplots(1, 1, figsize=(12, 9))
    # plt.imshow(combined, cmap='gray')
    # plt.title('Thresholded Gradient') #, fontsize=50)

    # plt.show()

    # Perspective transform

    img_size = gray.shape[::-1]

    s = 30
    l = 10

    src= np.float32([[540, 400],
                     [210, 580.],
                     [1050, 580.],
                     [690, 400]])

    # src = np.float32(   [[(img_size[0] / 2) - 55 - l, img_size[1] / 2 + 100],
    #                     [((img_size[0] / 6) - 10 - s), img_size[1]],
    #                     [((img_size[0] * 5 / 6) + 60 + s), img_size[1]],
    #                     [(img_size[0] / 2 + 55 + l), img_size[1] / 2 + 100]])

    dst = np.float32(   [[(img_size[0] / 4), 0],
                        [(img_size[0] / 4), img_size[1]],
                        [(img_size[0] * 3 / 4), img_size[1]],
                        [(img_size[0] * 3 / 4), 0]])

    # print(dst)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)

    # # Plot the result
    # f, (ax1, ax2) = plt.subplots(1, 2) #, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(combined, cmap = 'gray')
    # ax1.plot(src[0,0], src[0,1], '.')
    # ax1.plot(src[1,0], src[1,1], '.')
    # ax1.plot(src[2,0], src[2,1], '.')
    # ax1.plot(src[3,0], src[3,1], '.')
    # ax1.set_title('Original Image') #, fontsize=50)
    # ax2.imshow(warped, cmap='gray')
    # ax2.set_title('Thresholded Gradient') #, fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # plt.show()


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

    # Display the final results
    warpedimg = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
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
    # mark_size = 30
    # plt.imshow(warped, cmap = 'gray')
    # plt.plot(xl, y, 'o', color='red', markersize=mark_size)
    # plt.plot(xr, y, 'o', color='blue', markersize=mark_size)
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
    # print('Radius of Curvature =', curvature, 'm')



    # Overlay
    midx = warped.shape[1]/2
    # print('midx =', midx)

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
        position = 'left'
    else:
        position = 'right'
    cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), position),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    # plt.figure()
    # plt.imshow(color_warp)
    # plt.figure()

    # plt.figure()
    # plt.imshow(gradx, cmap = 'gray')
    # plt.title('gradx')

    # plt.figure()
    # plt.imshow(grady, cmap = 'gray')
    # plt.title('grady')

    # plt.figure()
    # plt.imshow(mag_binary, cmap = 'gray')
    # plt.title('mag_binary')

    # plt.figure()
    # plt.imshow(dir_binary, cmap = 'gray')
    # plt.title('dir_binary')







    # plt.figure()
    # plt.imshow(result)
    # plt.title('result')
    # plt.imshow(warped, cmap = 'gray')


    #plt.imshow(result)



    # plt.show()
    cv2.destroyAllWindows()

    return result
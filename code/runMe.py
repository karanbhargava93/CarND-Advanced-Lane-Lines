import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from functions import *

line = Line()

white_output = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4", audio = False)
white_clip = clip1.fl_image(line.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(white_output))



# img1 = mpimg.imread('../test_images/test4.jpg') # test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # 

# new_img1 = line.process_image(img1)

# plt.figure()
# plt.imshow(new_img1)


# img2 = mpimg.imread('../test_images/test4.jpg') # test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # 

# new_img2 = line.process_image(img2)

# plt.figure()
# plt.imshow(new_img2)

# # Original image
# img = mpimg.imread('../test_images/test7.jpg') # test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # 

# new_img = line.process_image(img)
# plt.figure()
# plt.imshow(new_img)

# # Original image
# img3 = mpimg.imread('../test_images/test7.jpg') # test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # 

# new_img3 = line.process_image(img3)
# plt.figure()
# plt.imshow(new_img3)

# plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
# from moviepy.video.io import VideoFileClip
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from functions import *

# white_output = 'output_video.mp4'
# clip1 = VideoFileClip("project_video.mp4", audio = False)
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

# HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(white_output))

# Original image
img = mpimg.imread('../test_images/test5.jpg') # test4.jpg') # straight_lines1.jpg') # test1.jpg') # straight_lines2.jpg') # 

new_img = process_image(img)

plt.imshow(new_img)
plt.show()
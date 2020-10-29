import numpy as np
import cv2
import matplotlib.pyplot as plt
# % matplotlib
# inline  # if you are running this code in jupyter notebook

img = cv2.imread('/path_to_image/opencv-logo.png', 0)  # reads image 'opencv-logo.png' as grayscale
plt.imshow(img, cmap='gray')
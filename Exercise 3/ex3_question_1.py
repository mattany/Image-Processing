import os
import sys

import cv2
import numpy as np
from skimage.filters import rank

# Load the Image
path_to_image = sys.argv[1]
img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)


# Equalization
selem = np.array([[1 for _ in range(3)] for i in range(3)])
img_eq = rank.equalize(img, selem=selem)
dirname = f"{os.path.split(path_to_image)[0]}"
filename = f"enhanced_{os.path.split(path_to_image)[1].split('.')[0]}.png"
cv2.imwrite(os.path.join(dirname, filename), img_eq)

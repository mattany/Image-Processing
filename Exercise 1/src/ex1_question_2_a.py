import numpy as np
import cv2
import sys

if __name__ == "__main__":
    path_to_image = sys.argv[1]
    img = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)
    cv2.imshow("original_image", img)
    cv2.waitKey(0)  # waits until a key is pressed

    img = np.array([[[v[2], v[1], v[0]] for v in r] for r in img])
    cv2.imwrite("reversed_colors.png", img)

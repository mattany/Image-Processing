import numpy as np
import cv2
import sys


def floyd_steinberg(mat):
    for i, row in enumerate(mat):
        for j, col in enumerate(row):
            old_value = mat[i][j]
            new_value = (mat[i][j] // 64) * 85
            mat[i][j] = new_value
            error = old_value - new_value
            if j < len(row) - 1:
                mat[i][j + 1] += (7 * error) / 16
            if i < len(mat) - 1 and j < len(row) - 1:
                mat[i + 1][j + 1] += error / 16
            if i < len(mat) - 1:
                mat[i + 1][j] += (5 * error) / 16
            if i < len(mat) - 1 and j > 0:
                mat[i + 1][j - 1] += (3 * error) / 16
    return mat


if __name__ == "__main__":
    path_to_image = sys.argv[1]
    img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    img = floyd_steinberg(img)
    cv2.imwrite("error_diffused_image2.png", img)
    cv2.waitKey(0)  # waits until a key is pressed

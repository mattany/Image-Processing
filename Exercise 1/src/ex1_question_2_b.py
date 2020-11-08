import numpy as np
import cv2
import sys
import os


def floyd_steinberg(mat):
    # convert to float matrix for more accurate calculations
    mat = mat.astype(np.float)

    for i, row in enumerate(mat):
        for j, col in enumerate(row):
            # choose the closest gray level and convert to it
            old_value = mat[i][j]
            levels = (0, 85, 170, 255)
            new_value = min(levels, key=lambda x: abs(old_value - x))
            error = old_value - new_value
            mat[i][j] = new_value

            # diffuse the error and clip to valid range
            if j < len(row) - 1:
                new_value = mat[i][j + 1] + (7 * error) / 16
                mat[i][j + 1] = clip(new_value)
            if i < len(mat) - 1 and j < len(row) - 1:
                new_value = mat[i + 1][j + 1] + error / 16
                mat[i + 1][j + 1] = clip(new_value)
            if i < len(mat) - 1:
                new_value = mat[i + 1][j] + (5 * error) / 16
                mat[i + 1][j] = new_value
            if i < len(mat) - 1 and j > 0:
                new_value = mat[i + 1][j - 1] + (3 * error) / 16
                mat[i + 1][j - 1] = clip(new_value)

    # convert back to 8 bit int matrix
    mat = mat.astype(np.uint8)
    return mat


def clip(new_value):
    if new_value > 255:
        new_value = 255
    elif new_value < 0:
        new_value = 0
    return new_value


if __name__ == "__main__":

    # Load, apply algorithm and write to new image
    path_to_image = sys.argv[1]
    img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    img = floyd_steinberg(img)
    dirname = f"{os.path.split(path_to_image)[0]}"
    filename = f"diffused_{os.path.split(path_to_image)[1].split('.')[0]}.png"
    cv2.imwrite(os.path.join(dirname, filename), img)

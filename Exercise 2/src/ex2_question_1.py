import numpy as np
import cv2
import sys
import os



filter = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]



def apply_filter(mat, filter):
    # convert the matrix to float
    mat = mat.astype(np.float)

    # construct sub matrices for convolution
    view_shape = tuple(np.subtract(mat.shape, filter.shape) + 1) + filter.shape
    strides = mat.strides + mat.strides
    sub_matrices = np.lib.stride_tricks.as_strided(mat, view_shape, strides)

    # convolve
    mat = np.einsum('ij,klij->kl', filter, sub_matrices)

    # clip out of bound values
    v_clip = np.vectorize(clip)
    mat = v_clip(mat)

    return mat.astype(np.uint8)


def clip(new_value):
    if new_value > 255:
        new_value = 255
    elif new_value < 0:
        new_value = 0
    return new_value


if __name__ == "__main__":
    path_to_image = sys.argv[1]
    img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

    img = apply_filter(img, np.asarray(filter))
    print(img)
    dirname = f"{os.path.split(path_to_image)[0]}"
    filename = f"filtered_{os.path.split(path_to_image)[1].split('.')[0]}.png"
    cv2.imwrite(os.path.join(dirname, filename), img)

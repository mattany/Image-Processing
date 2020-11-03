import numpy as np
import cv2
import sys
import os
import math

SATURATION = 1
INTENSITY = 2

HUE = 0


def construct_spectrum():
    hsi_mat = np.array([[[0, 1, 0.3] for i in range(720)] for i in range(100)])
    for row in hsi_mat:
        color = 7 / 4 * math.pi
        for pixel in row:
            pixel[HUE] = color
            color += math.pi / 360
            color %= 2 * math.pi
    mat = convert_to_rgb(hsi_mat)
    cv2.imwrite("./spectrum2.jpg", mat)


def convert_to_rgb(mat):
    for row in mat:
        for pixel in row:
            h, s, i = pixel[HUE], pixel[SATURATION], pixel[INTENSITY]
            if h < 2 / 3 * math.pi:
                b = i * (1 - s)
                r = i * (1 + ((s * math.cos(h)) / (math.cos((math.pi / 3) - h))))
                g = 3 * i - (b + r)
            elif h < 4 / 3 * math.pi:
                h -= 2 / 3 * math.pi
                r = i * (1 - s)
                g = i * (1 + ((s * math.cos(h)) / (math.cos((math.pi / 3) - h))))
                b = 3 * i - (r + g)
            else:
                h -= 4 / 3 * math.pi
                g = i * (1 - s)
                b = i * (1 + ((s * math.cos(h)) / (math.cos((math.pi / 3) - h))))
                r = 3 * i - (g + b)
            pixel[0], pixel[1], pixel[2] = 255 * r, 255 * g, 255 * b
    mat = mat.astype(np.uint8)
    return mat

construct_spectrum()

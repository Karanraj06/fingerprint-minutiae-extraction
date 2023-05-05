"""
The main goal of normalization is to reduce the variance of the gray 
level value along the ridges to facilitate subsequent processing steps
"""
import numpy as np


# Function to calculate normalized gray level for a pixel
def normalizePixel(val, m, v, m0, v0):
    """
    val = pixel value
    m = global image mean
    v = global image variance
    m0 = desired mean
    v0 = desired variance
    """
    x = np.sqrt((v0 * ((val - m) ** 2)) / v)
    if val < m:
        return m0 - x

    return m0 + x


# Function to perform normalization on an image
def normalize(img, m0, v0):
    m = np.mean(img)
    v = np.std(img) ** 2
    (w, h) = img.shape
    normalizedImg = img.copy()
    for i in range(w):
        for j in range(h):
            normalizedImg[i, j] = normalizePixel(img[i, j], m, v, m0, v0)

    return normalizedImg

"""
To facilitate extraction of minutiae the image must be skeletonized: a sequence of morphological
erosion operations will reduce the thickness of the striations until the latter is equal to one pixel
while maintaining the connectivity of the striations ( That is to say that the continuity of the
striaes must be respected, holes must not be inserted). While some papers use Rosenfeld algorithm for its
simplicity.
"""
import numpy as np
import cv2 as cv
from crossing_number import calculate_minutiaes
from skimage.morphology import skeletonize as skelt
from skimage.morphology import thin


def skeletonize(image_input):
    """
    Skeletonization reduces binary objects to 1 pixel wide representations.
    skeletonize works by making successive passes of the image. On each pass, border pixels are identified
    and removed on the condition that they do not break the connectivity of the corresponding object.
    """
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)
    skeleton = skelt(image)
    output[skeleton] = 255
    cv.bitwise_not(output, output)
    return output


def thinning_morph(image, kernel):
    """
    Thinning image using morphological operations
    """
    thining_image = np.zeros_like(image)
    img = image.copy()

    while 1:
        erosion = cv.erode(img, kernel, iterations=1)
        dilatate = cv.dilate(erosion, kernel, iterations=1)

        subs_img = np.subtract(img, dilatate)
        cv.bitwise_or(thining_image, subs_img, thining_image)
        img = erosion.copy()

        done = np.sum(img) == 0

        if done:
            break

    # shift down and compare one pixel offset
    down = np.zeros_like(thining_image)
    down[1:-1, :] = thining_image[0:-2,]
    down_mask = np.subtract(down, thining_image)
    down_mask[0:-2, :] = down_mask[1:-1,]
    cv.imshow("down", down_mask)

    # shift right and compare one pixel offset
    left = np.zeros_like(thining_image)
    left[:, 1:-1] = thining_image[:, 0:-2]
    left_mask = np.subtract(left, thining_image)
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv.imshow("left", left_mask)

    # combine left and down mask
    cv.bitwise_or(down_mask, down_mask, thining_image)
    output = np.zeros_like(thining_image)
    output[thining_image < 250] = 255

    return output

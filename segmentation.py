import numpy as np
import cv2 as cv


def normalise(img):
    return (img - np.mean(img)) / (np.std(img))


def create_segmented_and_variance_images(im, w, threshold=0.2):
    """
    Returns mask identifying the ROI. Calculates the standard deviation
    in each image block and threshold the ROI. It also normalises the intesity
    values of the image so that the ridge regions have zero mean, unit standard
    deviation.

    im: Image
    w: size of the block
    threshold: std threshold

    returns: segmented_image
    """
    (y, x) = im.shape
    threshold = np.std(im) * threshold

    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1] : box[3], box[0] : box[2]])
            image_variance[box[1] : box[3], box[0] : box[2]] = block_stddev

    # apply threshold
    mask[image_variance < threshold] = 0

    # smooth mask with a open/close morphological filter
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (w * 2, w * 2))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # normalize segmented image
    segmented_image *= mask
    im = normalise(im)
    mean_val = np.mean(im[mask == 0])
    std_val = np.std(im[mask == 0])
    norm_img = (im - mean_val) / (std_val)

    return segmented_image, norm_img, mask

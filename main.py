import cv2 as cv
from glob import glob
import os
import numpy as np
from poincare import calculate_singularities
from segmentation import create_segmented_and_variance_images
from normalization import normalize
from gabor_filter import gabor_filter
from frequency import ridge_freq
import orientation
from crossing_number import calculate_minutiaes
from tqdm import tqdm
from skeletonize import skeletonize


def f(input_img):
    # normalization -> orientation -> frequency -> mask -> filtering
    block_size = 16

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(
        normalized_img, block_size, 0.2
    )

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(
        segmented_img, mask, angles, W=block_size
    )

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(
        normim,
        mask,
        angles,
        block_size,
        kernel_size=5,
        minWaveLength=5,
        maxWaveLength=15,
    )

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    # minutias
    minutias = calculate_minutiaes(thin_image)

    # singularities
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

    # visualize pipeline stage by stage
    output_imgs = [
        input_img,
        normalized_img,
        segmented_img,
        orientation_img,
        gabor_img,
        thin_image,
        minutias,
        singularities_img,
    ]

    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate(
        [np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]
    ).astype(np.uint8)

    return results


if __name__ == "__main__":
    # open images
    img_dir = "./input/*"
    output_dir = "./output/"

    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path, 0) for img_path in images_paths])

    images = open_images(img_dir)

    # image pipeline
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images)):
        results = f(img)
        cv.imwrite(output_dir + str(i) + ".png", results)

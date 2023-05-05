import orientation
import math
import cv2 as cv
import numpy as np


def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    """
    cells = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):
        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop": (0, 0, 255), "delta": (0, 128, 255), "whorl": (255, 153, 255)}

    for i in range(3, len(angles) - 2):  # Y
        for j in range(3, len(angles[i]) - 2):  # x
            # mask any singularity outside of the mask
            mask_slice = mask[(i - 2) * W : (i + 3) * W, (j - 2) * W : (j + 3) * W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W * 5) ** 2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    cv.rectangle(
                        result,
                        ((j + 0) * W, (i + 0) * W),
                        ((j + 1) * W, (i + 1) * W),
                        colors[singularity],
                        3,
                    )

    return result

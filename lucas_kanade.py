import cv2
import numpy as np
from scipy.ndimage import filters


def harris(im, sigma=3):

    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), output= imx)

    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), output=imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet - Wtr


def get_harris_points(harris_im, min_distance = 10, threshold =0.1):

    corner_threshold = harris_im.max() * threshold
    harrisim_t = (harris_im > corner_threshold) * 1

    print(corner_threshold)
    print(harrisim_t)

    # get coordinates of candidates, the non-zero returns two separate arrays for the x and y dimensions. The transpose groups the corresponding x and y coordinates together.
    coords = np.array(harrisim_t.nonzero()).T

    # get values of candidates
    candidate_values = [harris_im[c[0], c[1]] for c in coords]

    print(candidate_values)

    if not candidate_values:
        print("No good corners detected in this image. Please select another set of points.")
    # sort candidates
    # index = argsort(candidate_values)[::-1]



img = cv2.imread('tunde.jpg') # np.random.randint(43, size=(3, 6))  # 6 columns, 3 rows.
img = img[:, :, 0] # remove the last dimension
harris_result = harris(img)
get_harris_points(harris_result)

import cv2
import numpy as np
from scipy.ndimage import filters


def harris(im, sigma=3):
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), output=imx)

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


def get_harris_points(harris_im, min_distance=10, threshold=0.1):
    corner_threshold = harris_im.max() * threshold
    harrisim_t = (harris_im > corner_threshold) * 1

    # get coordinates of candidates, the non-zero returns two separate arrays for the x and y dimensions.
    # The transpose groups the corresponding x and y coordinates together.
    coords = np.array(harrisim_t.nonzero()).T

    # get values of candidates
    candidate_values = [harris_im[c[0], c[1]] for c in coords]

    if not candidate_values:
        print("No good corners detected in this image. Please select another set of points.")

    # These are the indices of the candidate values if it was sorted.
    index = np.argsort(candidate_values)[::-1]

    # store the eligible locations
    eligible_locations = np.zeros(harris_im.shape)

    # factoring the minimum distance from the border.
    eligible_locations[min_distance: - min_distance, min_distance: -min_distance] = 1

    accepted_cords = []

    for i in index:
        if eligible_locations[coords[i, 0], coords[i, 1]]:
            # coords [i, 0] and [i,1] refer to the x and y coordinates for each index.
            accepted_cords.append(coords[i])

            # make all points without 'minimum distance' of a selected point ineligible.
            eligible_locations[(coords[i, 0] - min_distance):(coords[i, 0] + min_distance),
            (coords[i, 1] - min_distance):(coords[i, 1] + min_distance)] = 0

    return accepted_cords


img = cv2.imread('tundegray.png')  # np.random.randint(43, size=(3, 6))  # 6 columns, 3 rows.
img_2d = img[:, :, 0]  # remove the last dimension
harris_result = harris(img_2d)
good_corners = (get_harris_points(harris_result))

for cords in good_corners:
    img[cords[0] : cords[0], cords[1]: cords[1]] = [0, 0, 255]

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.imshow('frame', img)

k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import filters
# from main import Rect
import math, collections



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


def get_harris_points(harris_im, min_distance=10, threshold=0.5):
    corner_threshold = harris_im.max() * threshold
    harrisim_t = (harris_im > corner_threshold) * 1

    # get coordinates of candidates, the non-zero returns two separate arrays for the x and y dimensions.
    # The transpose groups the corresponding x and y coordinates together.
    coords = np.array(harrisim_t.nonzero()).T

    # get values of candidates
    candidate_values = [harris_im[c[0], c[1]] for c in coords]

    if not candidate_values:
        print("No good corners detected in this image. Please select another set of points.")
        return

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


# This is the method that implements the Lucas Kanade algorithm
def calc_optical_flow(im1, im2, corners: list, win=5):
    assert im1.shape == im2.shape

    x_kernel = np.array([[-1., 1.], [-1., 1.]])
    y_kernel = np.array([[-1., -1.], [1., 1.]])
    t_kernel = np.array([[1., 1.], [1., 1.]])

    win_size = math.floor(win / 2)

    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'  # This ensures that the convolution returns the same shape as the images
    boundary = 'symm'
    dx = signal.convolve2d(im1, x_kernel, boundary=boundary, mode=mode)
    dy = signal.convolve2d(im1, y_kernel, boundary=boundary, mode=mode)
    dt = signal.convolve2d(im2, t_kernel, boundary=boundary, mode=mode) + signal.convolve2d(im1, -t_kernel,
                                                                                            boundary=boundary,
                                                                                            mode=mode)
    u = np.zeros(len(corners))
    v = np.zeros(len(corners))
    # within window window_size * window_size

    for index, k in enumerate(corners):
        x = k[0]
        y = k[1]

        ix = dx[x - win_size: x + win_size, y - win_size: y + win_size + 1].flatten()
        iy = dy[x - win_size: x + win_size, y - win_size: y + win_size + 1].flatten()
        it = dy[x - win_size: x + win_size, y - win_size: y + win_size + 1].flatten()

        b = np.reshape(it, (it.shape[0], 1))  # get b here. Make b a column vector.
        a = np.vstack((ix, iy)).T  # get A here. Combine ix and iy into a matrix and transpose them.
        nu = np.matmul(np.linalg.pinv(a), b)  # get velocity here. Matrix inversion requires a square matrix but a isn't square.
                                              # This is why we use pinv.

        u[index] = nu[0]
        v[index] = nu[1]

    return u, v


measured_u, measured_v = 0, 0


# def run(rect: Rect, im1, im2):
#     im1_corners = im1[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x, 0]  # use the rows and the column specified
#     im1_2d = im1[:, :, 0]
#     im2_2d = im2[:, :, 0]
#     harris_result = harris(im1_corners)
#     good_corners = (get_harris_points(harris_result))
#
#     # TODO: Add condition for if there are no corners
#
#     # This fits the corners in the context of the whole image
#     scaled_corners = []
#
#     if isinstance(good_corners, collections.Iterable):
#         scaled_corners = [[corner[0] + rect.top_y, corner[1] + rect.top_x] for corner in good_corners]
#
#     u, v = calc_optical_flow(im1_2d, im2_2d, scaled_corners)
#
#     max_u = 0
#     max_v = 0
#
#     if u.any() and v.any():
#         max_u = math.floor(max(u, key=abs) * 0.1)
#         max_v = math.floor(max(v, key=abs) * 0.1)
#
#     new_rect = Rect(rect.top_x + max_v, rect.top_y + max_u, rect.bottom_x + max_v, rect.bottom_y + max_u)
#
#     return new_rect

    # cv2.rectangle(im1, (rect.top_x, rect.top_y), (rect.bottom_x, rect.bottom_y), (0, 255, 0), 3)
    # cv2.rectangle(im2, (new_react.top_x, new_react.top_y), (new_react.bottom_x, new_react.bottom_y), (0, 255, 0), 3)
    #
    # cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
    # cv2.imshow('frame1', im1)
    # cv2.imshow('frame2', im2)
    #
    # k = cv2.waitKey(0) & 0xFF
    #
    # if k == 27:
    #     cv2.destroyAllWindows()

#
# im1 = cv2.imread('1.jpg')
# im2 = cv2.imread('2.jpg')
#
# rect = Rect(470, 128, 900, 550)
#
# run(rect, im1, im2)


# Test code
img = cv2.imread('chessboard.jpg')  # np.random.randint(43, size=(3, 6))  #  5 rows, 7 columns.
img_2d = img[128: 550, 470:900, 0]  # remove the last dimension.img[rows, columns]
harris_result = harris(img_2d)
good_corners = (get_harris_points(harris_result))

for cords in good_corners:
    img[127 + cords[0]: 127 + cords[0] + 7, 469 + cords[1]: 469 + cords[1] + 7] = [0, 0, 255]

cv2.rectangle(img,(470,128) , (900, 550),(255,255,0),3) # (x- axis, y- axis)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.imshow('frame', img)

k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()

import cv2 as cv
import numpy as np
from scipy import signal
from scipy.ndimage import filters
import math
import collections
import harris
from collections import namedtuple


Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class LucasKanade:
    def __init__(self):
        self.u = 0
        self.v = 0
        self.harris = harris.Harris
        self.initial_frame = []
        self.initial_features =[]
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # This is the method that implements the Lucas Kanade algorithm
    def calc_optical_flow(self, im1, im2, corners: list, win=15):
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
            it = dt[x - win_size: x + win_size, y - win_size: y + win_size + 1].flatten()

            b = np.reshape(it, (it.shape[0], 1))  # get b here. Make b a column vector.
            a = np.vstack((ix, iy)).T  # get A here. Combine ix and iy into a matrix and transpose them.
            nu = np.matmul(np.linalg.pinv(a), b)  # get velocity here. Matrix inversion requires a square matrix but
            # a isn't square.
            # This is why we use pinv.

            u[index] = nu[0]
            v[index] = nu[1]

        return u,v

    def run(self, rect: Rect, im1, im2):
        im1_pure = im2[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x]
        im1_corners = im1[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x,
                      0]  # use the rows and the column specified

        im1_2d = im1[:, :, 0]
        im2_2d = im2[:, :, 0]

        if len(self.initial_frame) == 0:
            self.initial_frame = im1_2d
            pt = cv.goodFeaturesToTrack(im1_corners, mask=None, **self.feature_params)

            for i in range(len(pt)):
                pt[i][0][0] = pt[i][0][0] + rect.top_x
                pt[i][0][1] = pt[i][0][1] + rect.top_y

            self.initial_features = np.reshape(pt, (-1, 1, 2))

         # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(im1_2d, im2_2d, self.initial_features, None, **self.lk_params)

        p1 = np.reshape(p1, (-1, 1, 2))

        good_features = p1[st == 1]   # The shape of this thing is (x,y)
        self.initial_features = good_features

        return good_features





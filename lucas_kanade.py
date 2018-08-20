import cv2
import numpy as np
from scipy.signal import convolve2d
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

    # This is the method that implements the Lucas Kanade algorithm
    def calc_optical_flow(self, im1, im2, corners: list, win=15):
        assert im1.shape == im2.shape

        Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
        Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
        Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
        Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image

        Ix = (convolve2d(im1, Gx) + convolve2d(im2, Gx)) / 2  # smoothing in x direction

        Iy = (convolve2d(im1, Gy) + convolve2d(im2, Gy)) / 2  # smoothing in y direction
        It1 = convolve2d(im1, Gt1) + convolve2d(im2, Gt2)  # taking difference of two images using gaussian mask of all -1 and all 1

        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)

        A = np.zeros((2, 2))
        B = np.zeros((2, 1))

        for index, k in enumerate(corners):
            x, y = k

            A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)

            A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
            A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
            Ainv = np.linalg.pinv(A)

            B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
            prod = np.matmul(Ainv, B)

            u[y, x] = prod[0]
            v[y, x] = prod[1]

        u = u[u is not 0]
        v = v[v is not 0]

        return u,v

    def run(self, rect: Rect, im1, im2):
        im1_pure = im2[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x]
        im1_corners = im1[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x,
                      0]  # use the rows and the column specified
        im1_2d = im1[:, :, 0]
        im2_2d = im2[:, :, 0]

        # parameter to get features
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        features = cv2.goodFeaturesToTrack(im1_corners, mask=None,
                                          **feature_params)  # using opencv function to get feature for which we are plotting flow
        feature = np.int32(features)
        # print(feature)
        good_corners = np.reshape(feature, newshape=[-1, 2])

        # harris_result = self.harris.get_harris_value(im=im1_corners)
        # good_corners = (self.harris.get_harris_points(harris_result))

        if isinstance(good_corners, collections.Iterable):

            scaled_corners = [[corner[0] + rect.top_y, corner[1] + rect.top_x] for corner in good_corners]
            u, v = self.calc_optical_flow(im1_2d, im2_2d, scaled_corners)

            if u.any() and v.any():
                self.u = math.floor(np.array(u).mean())
                self.v = math.floor(np.array(v).mean())

        u, v = self.u, self.v
        self.u, self.v = 0, 0
        return u, v


import cv2 as cv
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
        self.initial_frame = []
        self.initial_features =[]
        self.initial_corners = []
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=5,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


    # This is the method that implements the Lucas Kanade algorithm
    def calc_optical_flow(self, im1, im2, corners: np.array, win=15):
        assert im1.shape == im2.shape

        dx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
        dy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
        dt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
        dt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


        Ix = (convolve2d(im1, dx) + convolve2d(im2, dx)) / 2  # smoothing in x direction

        Iy = (convolve2d(im1, dy) + convolve2d(im2, dy)) / 2  # smoothing in y direction
        It1 = convolve2d(im1, dt1) + convolve2d(im2,dt2)
        # taking difference of two images using gaussian mask of all -1 and all 1


        corners = np.reshape(corners, newshape=[-1,2])
        win_size = math.floor(win / 2)


        corners = np.int32(corners)

        u = np.ones(Ix.shape)
        v = np.ones(Ix.shape)
        A = np.zeros((2, 2))
        B = np.zeros((2, 1))

        status = np.zeros(corners.shape[0])

        new_corners = np.zeros_like(corners)

        # within window window_size * window_size
        for index, k in enumerate(corners):
            x = k[0]
            y = k[1]

            # Lucas Kanade imp
            A[0, 0] = np.sum((Ix[y - win_size:y + win_size, x - win_size:x + win_size]) ** 2)

            A[1, 1] = np.sum((Iy[y - win_size:y + win_size, x - win_size:x + win_size]) ** 2)
            A[0, 1] = np.sum(Ix[y - win_size:y + win_size, x - win_size:x + win_size] * Iy[y - win_size:y + win_size, x - win_size:x + win_size])
            A[1, 0] = np.sum(Ix[y - win_size:y + win_size, x - win_size:x + win_size] * Iy[y - win_size:y + win_size, x - win_size:x + win_size])
            Ainv = np.linalg.pinv(A)

            B[0, 0] = -np.sum(Ix[y - win_size:y + win_size, x - win_size:x + win_size] * It1[y - win_size:y + win_size, x - win_size:x + win_size])
            B[1, 0] = -np.sum(Iy[y - win_size:y + win_size, x - win_size:x + win_size] * It1[y - win_size:y + win_size, x - win_size:x + win_size])
            prod = np.matmul(Ainv, B)

            u[y, x] = prod[0]
            v[y, x] = prod[1]
            new_corners[index] = [np.int32(x + u[y,x]), np.int32(y + v[y,x])]

            if np.int32(x + u[y, x]) == x and np.int32(y + v[y, x]) == y:
                status[index] = 0
            else:
                status[index] = 1

        new_corners = new_corners[status == 1]

        return new_corners

    def run(self, rect: Rect, im1, im2):
        im1_pure = im2[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x]
        im1_corners = im1[rect.top_y: rect.bottom_y, rect.top_x: rect.bottom_x,
                      0]  # use the rows and the column specified



        im1_2d = im1[:, :, 0]
        im2_2d = im2[:, :, 0]

        # harris_result = self.harris.get_harris_value(im1_corners)

        if len(self.initial_frame) == 0:
            self.initial_frame = im1_2d
            pt = cv.goodFeaturesToTrack(im1_corners, mask=None, **self.feature_params, useHarrisDetector=True)
            #  self.initial_corners = self.harris.get_harris_points(harris_result) # Corners here are in form row,column

            for i in range(len(pt)):
                pt[i][0][0] = pt[i][0][0] + rect.top_x
                pt[i][0][1] = pt[i][0][1] + rect.top_y

            self.initial_features = np.reshape(pt, (-1, 1, 2))  # Get the first set of features

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(im1_2d, im2_2d, self.initial_features, None, **self.lk_params)
        # new_corners = self.calc_optical_flow(im1_2d, im2_2d, self.initial_features)

        p1 = np.reshape(p1, (-1, 1, 2))

        good_features = p1[st == 1]   # The shape of this thing is (x,y)

        self.initial_features = good_features
        # self.initial_corners = new_corners

        return good_features



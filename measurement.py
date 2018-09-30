import lucas_kanade
import background_subtractor
import kalman
import numpy as np
from collections import namedtuple
import math
import cv2

Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class Measurement:
    def __init__(self, rect):
        self.rect = rect
        self.im1 = []
        self.im2 = []
        self.lk = lucas_kanade.LucasKanade()
        self.bgsubtractor = background_subtractor.BackgroundSubtractor()
        self.measured = []
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def update(self, im1, im2, measured_rect):
        self.rect = measured_rect
        self.im1 = im1
        self.im2 = im2

    def measure(self):
        self.measured = self.lk.run(self.rect, self.im1, self.im2)

    def run(self):
        self.measure()

        x = []
        y = []

        for i, value in enumerate(self.measured):
            x.append(value[0])
            y.append(value[1])

        x = np.array(x)
        y = np.array(y)

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        top_x = math.ceil(x_min)
        top_y = math.ceil(y_min)
        bottom_x = math.ceil(x_max)
        bottom_y = math.ceil(y_max)

        self.bgsubtractor.run(self.im2)

        if len(self.bgsubtractor.contours) > 0:
            rect_coordinates = self.bgsubtractor.get_suitable_rectangles(self.rect)

            if len(rect_coordinates) > 0:
                self.rect = self.bgsubtractor.get_best_candidate(rect_coordinates, self.measured)
        else:
            self.rect = Rect(top_x, top_y, bottom_x, bottom_y)


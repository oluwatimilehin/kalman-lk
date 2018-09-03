import lucas_kanade
import background_subtractor
import kalman
import numpy as np
from collections import namedtuple
import math
import cv2

Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class Tracker:
    def __init__(self, rect):
        self.rect = rect
        self.im1 = []
        self.im2 = []
        self.lk = lucas_kanade.LucasKanade()
        self.bgsubtractor = background_subtractor.BackgroundSubtractor()
        self.measured = []
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def update(self, im1, im2, rect):
        self.rect = rect
        self.im1 = im1
        self.im2 = im2
        self.measure()

    def measure(self):
        self.measured = self.lk.run(self.rect, self.im1, self.im2)

    def run(self):
        self.measure()

        self.bgsubtractor.run(self.im2)

        if len(self.bgsubtractor.contours) > 0:
            rect_coordinates = self.bgsubtractor.get_suitable_rectangles(self.rect)

            if len(rect_coordinates) > 0:
                self.rect = self.bgsubtractor.get_best_candidate(rect_coordinates, self.measured)



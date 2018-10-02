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
        self.old_rect = rect
        self.max_x = 0
        self.max_y = 0

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
        bottom_y = math.ceil(y_max)
        bottom_x = math.ceil(x_max)
        top_y = math.ceil(y_min)

        if len(self.bgsubtractor.contours) > 0:
            rect_coordinates = self.bgsubtractor.get_suitable_rectangles(self.rect)

            if len(rect_coordinates) > 0:
                self.rect = self.bgsubtractor.get_best_candidate(rect_coordinates, self.measured)
                rect = self.rect

                if self.rect.bottom_x - self.rect.top_x > self.max_x:
                    self.max_x = self.rect.bottom_x - self.rect.top_x

                if self.rect.bottom_y - self.rect.top_y > self.max_y:
                    self.max_y = self.rect.bottom_y - self.rect.top_y

                if rect.bottom_x - rect.top_x > 80 or rect.bottom_y - rect.top_y > 140:
                    print("Occlusion detected")
                    self.match_template()

        else:
            self.rect = Rect(top_x - 20, top_y - 30, bottom_x + 20, bottom_y -20)

        self.old_rect = self.rect

    def match_template(self):
        im1_2d = self.im1[:, :, 0]
        im2_2d = self.im2[:, :, 0]
        template = im1_2d[self.old_rect.top_y: self.old_rect.bottom_y, self.old_rect.top_x: self.old_rect.bottom_x]
        w, h = template.shape[::-1]

        var = 50

        res = cv2.matchTemplate(im2_2d[self.rect.top_y - var : self.rect.bottom_y + var, self.rect.top_x - var:self.rect.bottom_x + var], template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        x = top_left[0] + self.rect.top_x - var
        y= top_left[1] + self.rect.top_y - var

        top_left = (x,y)

        bottom_right = (x + w, y + h)

        self.rect = Rect(top_left[0], top_left[1], bottom_right[0], bottom_right[1])






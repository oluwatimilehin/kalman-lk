import lucas_kanade
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
        rect_variance = 500
        fgmask = self.fgbg.apply(self.im2) #[self.rect.top_y - rect_variance:self.rect.bottom_y + rect_variance,
                                 #self.rect.top_x - rect_variance:self.rect.bottom_x+rect_variance, 2])
        ret, thresh = cv2.threshold(fgmask, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rect_coordinates = []
        best_rect = [(self.rect.top_x, self.rect.top_y), (self.rect.bottom_x, self.rect.bottom_y)]

        count_dict = {}
        if len(contours) > 0:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 5:
                    bottom_x, bottom_y = x + w, y + h
                    if x >= best_rect[0][0] - 50 and bottom_x <= best_rect[1][0] + 100:
                        rect_coordinates.append([(x, y), (bottom_x, bottom_y)])

        if len(rect_coordinates) is not 0:
            for index, cord in enumerate(rect_coordinates):
                top_point = cord[0]
                bottom_point = cord[1]

                count = 0
                for point in self.measured:
                    if top_point[0] <= point[0] <= bottom_point[0] or top_point[1] <= point[1] <=bottom_point[1]:
                        count += 1

                count_dict[index] = count

            max_index = 0
            max_val = 0

            for k, v in count_dict.items():
                if v > max_val:
                    max_val = v
                    max_index = k

            best_rect = rect_coordinates[max_index]

        self.rect = Rect(best_rect[0][0], best_rect[0][1], best_rect[1][0], best_rect[1][1])

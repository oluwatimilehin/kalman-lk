import cv2
from collections import namedtuple
import math
import sys

Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class BackgroundSubtractor:
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        self.contours = []

    def run(self, im):
        fgmask = self.fgbg.apply(im)  # [self.rect.top_y - rect_variance:self.rect.bottom_y + rect_variance,
        # self.rect.top_x - rect_variance:self.rect.bottom_x+rect_variance, 2])
        ret, thresh = cv2.threshold(fgmask, 127, 255, 0)
        im2, self.contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    def get_suitable_rectangles(self, rect: Rect):
        rect_coordinates = []
        if len(self.contours) > 0:
            suitable_contours = [cnt for cnt in self.contours if len(cnt) > 0] # This is to reduce computation time

            rects = []
            for i in self.contours:
                curr_rect = cv2.boundingRect(i)

                bottom_x, bottom_y=curr_rect[0] + curr_rect[2], curr_rect[1] + curr_rect[3]
                rects.append([curr_rect[0], curr_rect[1], bottom_x, bottom_y])

            for cnt in suitable_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 10 and h > 10:
                    bottom_x, bottom_y = x + w, y + h
                    if (rect.top_x >= x and rect.bottom_x <= bottom_x) or \
                            (rect.top_y >= y and rect.bottom_y <= bottom_y) or \
                            (x >= rect.top_x and bottom_x <= rect.bottom_x) or \
                            (y >= rect.top_x and bottom_y < rect.bottom_y):
                            rect_coordinates.append(Rect(x, y,bottom_x,bottom_y))

        return rect_coordinates

    @staticmethod
    def get_best_candidate(rect_coordinates: list, points, previous_rect: Rect):
        centre_x = (previous_rect.top_x + previous_rect.bottom_x) / 2
        centre_y = (previous_rect.top_y + previous_rect.bottom_y) / 2

        min_index = 0
        min_val = sys.maxsize

        for index, rect in enumerate(rect_coordinates):
            x = (rect.top_x + rect.bottom_x) /2
            y = (rect.top_y + rect.bottom_y) /2

            euc_x = math.pow((centre_x - x), 2)
            euc_y = math.pow((centre_y-y), 2)

            euc_dist = math.sqrt(euc_x + euc_y)

            if euc_dist < min_val:
                min_val = euc_dist
                min_index = index

        best_rect = rect_coordinates[min_index]
        #
        # if abs(best_rect.top_x - previous_rect.top_x) > 5 and abs(best_rect.top_y - previous_rect.top_y) > 5:
        #     print("Why")

        return best_rect

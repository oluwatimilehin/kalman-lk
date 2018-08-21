import lucas_kanade
import kalman
import numpy as np
from collections import namedtuple
import math

Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class Tracker:
    def __init__(self, rect):
        self.rect = rect
        self.im1 = []
        self.im2 = []
        self.lk = lucas_kanade.LucasKanade()
        self.measured = []

    def update(self, im1, im2, rect):
        self.rect = rect
        self.im1 = im1
        self.im2 = im2
        self.measure()

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

        self.rect = Rect(top_x, top_y, bottom_x, bottom_y)

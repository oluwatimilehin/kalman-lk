import lucas_kanade
import kalman
import numpy as np
from collections import namedtuple
import math

Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class Tracker:
    def __init__(self, rect):
        self.rect = rect
        self.lk = lucas_kanade.LucasKanade()
        self.kalman = kalman.Kalman(self.rect)
        self.im1 = []
        self.im2 = []
        self.measured = []

    def update(self, im1, im2, rect):
        self.rect = rect
        self.im1 = im1
        self.im2 = im2

    def measure(self):
        u, v = self.lk.run(self.rect, self.im1, self.im2)
        self.measured = np.array([u, v]).T

    def run(self):
        self.kalman.predict()
        self.measure()
        self.kalman.update(self.measured)

        top_x = math.ceil(self.kalman.x[0])
        top_y = math.ceil(self.kalman.x[2])
        bottom_x = math.ceil(self.rect.bottom_x + (self.kalman.x[1]))
        bottom_y = math.ceil(self.rect.bottom_y + (self.kalman.x[3]))

        self.rect = Rect(top_x, top_y, bottom_x, bottom_y)

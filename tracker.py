import lucas_kanade
import kalman
import kalmane
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
        # self.kalman = kalman.Kalman(self.rect, self.measured)
        self.kalman = kalmane.Kalman_E(self.rect, self.measured)

    def measure(self):
        u, v = self.lk.run(self.rect, self.im1, self.im2)

        new_x = ((self.rect.top_x + self.rect.bottom_x) /2) + u
        new_y = (self.rect.top_y + self.rect.bottom_y) / 2 + v

        self.measured = np.array([u, v]).T

    def run(self):
        self.kalman.predict()
        self.measure()
        self.kalman.update(self.measured)

        top_x = math.ceil(self.rect.top_x + self.kalman.player_f.x[1])
        top_y = math.ceil(self.rect.top_y + self.kalman.player_f.x[3])
        bottom_x = math.ceil(self.rect.bottom_x + (self.kalman.player_f.x[1]))
        bottom_y = math.ceil(self.rect.bottom_y + (self.kalman.player_f.x[3]))

        self.rect = Rect(top_x, top_y, bottom_x, bottom_y)

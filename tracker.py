import lucas_kanade
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
        self.new_x = 0
        self.new_y = 0
        self.kalman = kalmane.Kalman_E(self.rect)

    def update(self, im1, im2, rect):
        self.rect = rect
        self.im1 = im1
        self.im2 = im2
        self.measure()
        # self.kalman = kalman.Kalman(self.rect, self.measured)

    def measure(self):
        u, v = self.lk.run(self.rect, self.im1, self.im2)

        self.new_x = ((self.rect.top_x + self.rect.bottom_x) /2) + u
        self.new_y = ((self.rect.top_y + self.rect.bottom_y) / 2) + v

        self.measured = np.array([self.new_x, self.new_y]).T

    def run(self):
        self.kalman.predict()
        self.measure()
        self.kalman.update(self.measured)

        var_top_x = self.rect.top_x - self.new_x
        var_top_y = self.rect.top_y - self.new_y
        var_bottom_x = self.rect.bottom_x - self.new_x
        var_bottom_y = self.rect.bottom_y - self.new_y

        top_x = math.floor(self.kalman.player_f.x[0] + var_top_x)
        top_y = math.floor(self.kalman.player_f.x[2] + var_top_y)
        bottom_x = math.floor(self.kalman.player_f.x[0] + var_bottom_x)
        bottom_y = math.floor(self.kalman.player_f.x[2] + var_bottom_y)

        self.rect = Rect(top_x, top_y, bottom_x, bottom_y)

import kalmane
import numpy as np
from collections import namedtuple
import math
import measurement

Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class Tracker:
    def __init__(self, rect):
        self.rect = rect
        self.im1 = []
        self.im2 = []
        self.measurement = measurement.Measurement(rect)
        self.measured = []
        self.measured_rect = rect
        self.new_x = 0
        self.new_y = 0
        self.kalman = kalmane.Kalman_E(self.rect)

    def update_params(self, im1, im2, rect, measured_rect):
        self.rect = rect
        self.im1 = im1
        self.im2 = im2
        self.measurement.update(im1, im2, measured_rect)

    def measure(self):
        self.measurement.run()
        self.rect = self.measurement.rect
        self.measured_rect = self.measurement.rect
        self.new_x = math.floor((self.rect.top_x + self.rect.bottom_x) / 2)
        self.new_y = math.floor((self.rect.top_y + self.rect.bottom_y) / 2)

        self.measured = np.array([self.new_x, self.new_y]).T

    def run(self):
        self.kalman.predict()
        self.measure()
        self.kalman.update(self.measured)

        # This section is used to determine the position of the rectangle from the centre point being tracked
        var_top_x = self.rect.top_x - self.new_x
        var_top_y = self.rect.top_y - self.new_y
        var_bottom_x = self.rect.bottom_x - self.new_x
        var_bottom_y = self.rect.bottom_y - self.new_y

        if self.measured_rect.top_x <= self.kalman.player_f.x[0] <= self.measured_rect.bottom_x:
            if self.measured_rect.top_y <= self.kalman.player_f.x[2] <= self.measured_rect.bottom_y:
                # print("Use bounding")
                return

        top_x = math.floor(self.kalman.player_f.x[0] + var_top_x)
        top_y = math.floor(self.kalman.player_f.x[2] + var_top_y)
        bottom_x = math.floor(self.kalman.player_f.x[0] + var_bottom_x)
        bottom_y = math.floor(self.kalman.player_f.x[2] + var_bottom_y)

        # print("Not bounding")
        self.rect = self.measured_rect #Rect(top_x, top_y, bottom_x, bottom_y)

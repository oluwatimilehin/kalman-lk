from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from filterpy.common import Saver
from main import Rect


class Kalman_E(object):
    def __init__(self, rect: Rect,dt=1):
        self.player_f= KalmanFilter(dim_x=4, dim_z=2)
        center_x = (rect.top_x + rect.bottom_x) / 2
        center_y = (rect.top_y + rect.bottom_y) /2
        self.player_f.x = np.array([center_x, 0, center_y, 0])
        self.player_f.F = np.array([[1, dt, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, dt],
                               [0, 0, 0, 1]])
        self.player_f.P = np.diag([0.5, 0.5, 0.5, 0.5])
        self.player_f.R = np.diag([50, 50])
        self.player_f.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        self.player_f.Q = Q_discrete_white_noise(dim=4, dt=1, var=900)
        self.s = Saver(self.player_f)

    def predict(self):
        self.player_f.predict()

    def update(self, measured):
        self.player_f.update(measured)
        self.s.save()
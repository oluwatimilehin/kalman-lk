from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from filterpy.common import Saver
from main import Rect


class Kalman_E(object):
    def __init__(self, rect: Rect, measured, dt=0.1):
        self.player_f= KalmanFilter(dim_x=4, dim_z=2)
        self.player_f.x = np.array([rect.top_x, measured[0], rect.top_y, measured[1]])
        self.player_f.F = np.array([[1, dt, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, dt],
                               [0, 0, 0, 1]])
        self.player_f.P = np.diag([1, 3.2, 1, 3.2])
        self.player_f.R = np.diag([0.2, 0.2])
        self.player_f.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        self.player_f.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=500)
        self.s = Saver(self.player_f)

    def predict(self):
        self.player_f.predict()

    def update(self, measured):
        self.player_f.update(measured)
        self.s.save()
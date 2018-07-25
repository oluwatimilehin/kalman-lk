import numpy as np
from numpy import dot
from main import Rect
from scipy.linalg import inv


class Kalman(object):
    def __init__(self, rect: Rect, measured, dt=0.1):
        self.x = np.array([rect.top_x, measured[0], rect.top_y, measured[1]]).T
        self.P = np.diag([1, 3.2, 1, 3.2])
        self.Q = np.array([[3, 0.5, 0, 0],
                           [0.5, 3, 0, 0],
                           [0, 0, 3, 0.5],
                           [0, 0, 0.5, 3]])
        self.H = np.array([[0, 1, 0, 0],
                           [0, 0, 0, 1]])
        self.R = np.diag([0.2, 0.2])
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])

    def predict(self):
        self.x = dot(self.F, self.x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

    def update(self, z):
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S))
        y = z - dot(self.H, self.x)
        self.x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
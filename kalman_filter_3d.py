import numpy as np


class KalmanTracker3D:
    def __init__(self):
        self.active = False  # Track if we've seen this ball
        self.frames_missing = 0  # Count frames where ball wasn't detected

        # Initialize Kalman filter parameters
        self.X = np.zeros((9, 1))  # State
        self.P = np.eye(9) *100   # Uncertainty
        self.F = np.array([
            [1, 1,0.5,0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1,0.5,0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1,0.5],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0 ,1, 0, 0]
        ])
        self.R = np.eye(3)*1e2
        self.u = np.zeros((9, 1))

    def update(self, measurement):
        # Predict
        self.X = self.F @ self.X + self.u
        self.P = self.F @ self.P @ self.F.T

        if measurement is not None:
            # Update
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            Y = np.expand_dims(measurement,-1) - self.H @ self.X
            self.X = self.X + K @ Y
            self.P = (np.eye(9) - K @ self.H) @ self.P
    
    def get_position(self):
        return (self.X[0, 0], self.X[3, 0], self.X[6, 0])
    

    

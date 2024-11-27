import cv2
import numpy as np


def update(x, P, Z, H, R):
    # Measurement update
    I = np.eye(6)
    
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    Y = Z - H @ x
    x = x + K @ Y
    P = (I - K @ H) @ P
    return x, P

def predict(x, P, F, u):
    # Prediction step
    x = F @ x + u
    P = F @ P @ F.T
    return x, P

class KalmanTracker:
    def __init__(self):
        self.active = False  # Track if we've seen this ball
        self.frames_missing = 0  # Count frames where ball wasn't detected
        
        # Initialize Kalman filter parameters
        self.X = np.zeros((6, 1))  # State
        self.P = np.eye(6) * 100   # Uncertainty
        self.F = np.array([
            [1, 1, 0.5, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0.5],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        self.R = np.eye(2)
        self.u = np.zeros((6, 1))

    def update(self, measurement):
        # Predict
        self.X = self.F @ self.X + self.u
        self.P = self.F @ self.P @ self.F.T

        if measurement is not None:
            self.active = True
            self.frames_missing = 0
            # Update
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            Y = measurement - self.H @ self.X
            self.X = self.X + K @ Y
            self.P = (np.eye(6) - K @ self.H) @ self.P
        else:
            self.frames_missing += 1
            if self.frames_missing > 30:  # Reset if ball is lost for too long
                self.active = False

        return self.X
    
    def get_position(self):
        # print("XXXX")
        # print(self.X)
        # print("position x")
        # print(self.X[0, 0])
        # print("position y")
        # print(self.X[3, 0])
        
        return (int(self.X[0, 0]), int(self.X[3, 0]))
    
    def get_velocity(self):
        return (int(self.X[1, 0]), int(self.X[4, 0]))
    
    def get_predicted_position(self):
        pos = self.get_position()
        vel = self.get_velocity()
        return (pos[0] + vel[0], pos[1] + vel[1])
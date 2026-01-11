import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1/120, process_var=1e-5, meas_var=1e-2):
        self.dt = dt
        self.A = np.array([[1, dt], [0, 1]])  # state transition
        self.H = np.array([[1, 0]])
        self.Q = process_var * np.eye(2)       # process noise
        self.R = meas_var                     # measurement noise
        self.P = np.eye(2)
        self.x = np.zeros(2)                  # initial state

    def step(self, z):
        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Update
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
        self.x = x_pred + K.flatten() * (z - self.H @ x_pred)
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x.copy()

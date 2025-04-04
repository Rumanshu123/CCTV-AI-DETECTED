import cv2
import numpy as np

class KalmanFaceTracker:
    def __init__(self):
        # State: [x, y, width, height, vx, vy, vw, vh]
        self.kalman = cv2.KalmanFilter(8, 4)
        
        # Transition matrix (assuming constant velocity)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = 1e-4 * np.eye(8, dtype=np.float32)
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = 1e-1 * np.eye(4, dtype=np.float32)
        
        # Initial state
        self.kalman.statePost = np.zeros((8, 1), dtype=np.float32)
        
    def update(self, detection):
        # Detection is [x, y, width, height]
        measurement = np.array([
            [detection[0]],
            [detection[1]],
            [detection[2]],
            [detection[3]]
        ], dtype=np.float32)
        
        # Predict and correct
        self.kalman.predict()
        self.kalman.correct(measurement)
        
        # Get updated state
        state = self.kalman.statePost
        
        return [state[0, 0], state[1, 0], state[2, 0], state[3, 0]]
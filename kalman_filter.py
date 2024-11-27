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

def detect_balls(frame, lower_color, upper_color, color_name):
    """Detect balls of a specified color in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for the specified color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find circles in the mask
    edges = cv2.Canny(mask, 75, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                 param1=150, param2=12, minRadius=70, maxRadius=90)
    
    return circles, mask

def detect_balls(frame, lower_colors, upper_colors, color_name):
    """Detect balls of a specified color in the frame using optimized parameters."""
    # Resize frame to reduce processing time
    scale = 0.5
    small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Convert to HSV once
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    combined_mask = None
    # Handle multiple color ranges (e.g., for red color that wraps around hue)
    for lower, upper in zip(lower_colors, upper_colors):
        mask = cv2.inRange(hsv, lower, upper)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Optimize morphological operations with smaller kernel
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find circles with optimized parameters
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50 * scale,  # Adjusted for scaled image
        param1=150,
        param2=11,
        minRadius=int(60 * scale),  # Adjusted for scaled image
        maxRadius=int(100 * scale)   # Adjusted for scaled image
    )
    
    # Scale circles back to original size if found
    if circles is not None:
        circles[0, :, :2] /= scale
        circles[0, :, 2] /= scale
        
    return circles

class KalmanTracker:
    def __init__(self):
        self.active = False  # Track if we've seen this ball
        self.frames_missing = 0  # Count frames where ball wasn't detected
        
        # Initialize Kalman filter parameters
        self.X = np.zeros((6, 1))  # State
        self.P = np.eye(6) * 1000   # Uncertainty
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

def draw_tracker_info(frame, tracker, color_dict):
    """Draw tracking information for a single ball."""
    if not tracker.active:
        return

    X = tracker.X
    color = color_dict[tracker.color_name]
    
    # Draw current position
    pos = (int(X[0, 0]), int(X[3, 0]))
    cv2.circle(frame, pos, 5, color, -1)
    
    # Draw velocity vector
    vel = (int(X[1, 0]), int(X[4, 0]))
    next_pos = (pos[0] + vel[0], pos[1] + vel[1])
    cv2.arrowedLine(frame, pos, next_pos, color, 2)
    
    # Draw text
    text_color = (255, 255, 255)
    texts = [
        f"{tracker.color_name} Pos: ({X[0,0]:.1f}, {X[3,0]:.1f})",
        f"Vel: ({X[1,0]:.1f}, {X[4,0]:.1f})",
        f"Acc: ({X[2,0]:.1f}, {X[5,0]:.1f})"
    ]
    
    base_y = {'red': 30, 'yellow': 120, 'blue': 210}[tracker.color_name]
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (10, base_y + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

def main():
    # Color configurations
    color_ranges = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        'blue': [(np.array([90, 100, 100]), np.array([130, 255, 255]))]
    }
    
    # BGR colors for visualization
    color_dict = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'blue': (255, 0, 0)
    }

    # Initialize trackers
    trackers = {
        color: KalmanTracker(color) 
        for color in ['red', 'yellow', 'blue']
    }
    
    # Video capture
    # cap = cv2.VideoCapture('rolling_ball.mp4')
    cap = cv2.VideoCapture('rolling_ball_challenge.mp4')
    if not cap.isOpened():
        print("Cannot open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # Process each color
        for color, ranges in color_ranges.items():
            # Detect balls
            circles = detect_balls(
                frame,
                [range[0] for range in ranges],
                [range[1] for range in ranges],
                color
            )
            
            # Update Kalman filter
            measurement = None
            if circles is not None:
                # Draw detected circle
                x, y, r = circles[0][0]
                cv2.circle(display_frame, (int(x), int(y)), int(r), color_dict[color], 2)
                measurement = np.array([[x], [y]])
            
            # Update tracker
            trackers[color].update(measurement)
            
            # Draw tracker information
            draw_tracker_info(display_frame, trackers[color], color_dict)

        cv2.imshow('Multi-Color Tracking', display_frame)
        
        framerate = 60
        delay = int(1000 / framerate)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
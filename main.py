import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import deque
import json
from typing import Optional, Dict

# Global variables
home_position = None
home_rotation_matrix = None
camera_position = None
camera_rotation_matrix = None
real_position = None
real_rotation = None
distance = None
FILTER_WINDOW_SIZE = 10  # Adjust this value to change smoothing (higher = smoother but more latency)
position_history = deque(maxlen=FILTER_WINDOW_SIZE)
rotation_history = deque(maxlen=FILTER_WINDOW_SIZE)

def load_camera_data(file_path):
    """Load camera matrix and distortion coefficients from a file."""
    with np.load(file_path) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    return camera_matrix, dist_coeffs

def load_color_calibration(file_path):
    """Load HSV color ranges for red bricks from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            lower_red1 = np.array(data['lower_red1'])
            upper_red1 = np.array(data['upper_red1'])
            lower_red2 = np.array(data['lower_red2'])
            upper_red2 = np.array(data['upper_red2'])
            return lower_red1, upper_red1, lower_red2, upper_red2
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print(f"Warning: Could not load calibration from {file_path}. Using default values.")
        return np.array([0, 100, 100]), np.array([10, 255, 255]), np.array([170, 100, 100]), np.array([179, 255, 255])

def detect_red_bricks(frame, model, lower_red1, upper_red1, lower_red2, upper_red2):
    """Detect red bricks in the frame using YOLO and color filtering with provided HSV ranges."""
    # Run YOLO detection
    results = model(frame)
    output_image = frame.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            
            # Convert ROI to HSV color space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create masks for red color using provided ranges
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Calculate percentage of red pixels
            red_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Classify as red brick if more than 50% of pixels are red
            if total_pixels > 0 and red_pixels / total_pixels > 0.5:
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, 'Red Brick', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image

# Load YOLO model
model = YOLO('yolov8n.pt')  # Ensure 'yolov8n.pt' is available or specify the correct path

# Load camera data
camera_matrix, dist_coeffs = load_camera_data('calibration_data_1080.npz')

# Load color calibration
lower_red1, upper_red1, lower_red2, upper_red2 = load_color_calibration('calibration.json')
print(f"Using color calibration: lower_red1={lower_red1}, upper_red1={upper_red1}, lower_red2={lower_red2}, upper_red2={upper_red2}")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect red bricks using the loaded calibration
    output_image = detect_red_bricks(frame, model, lower_red1, upper_red1, lower_red2, upper_red2)
    
    # Display the output
    cv2.imshow('Detected Red Bricks', output_image)
    
    # Check for 'q' key to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import deque
import json
from typing import Optional, Dict
import brick_detection
import arucotag_detection

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

if __name__ == "__main__":
    # Load YOLO model
    model = YOLO('yolo11s.pt')  # Ensure 'yolov8n.pt' is available or specify the correct path

    # Load camera data
    camera_matrix, dist_coeffs = load_camera_data('calibration_data_1080.npz')

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect red bricks using the loaded calibration
        output_image = brick_detection.detect_red_bricks(frame, model)
        
        # detect aruco tags
        output_image = arucotag_detection.detect_aruco_markers(output_image, camera_matrix, dist_coeffs)

        # Display the output
        cv2.imshow('Detected Red Bricks', output_image)
        
        # Check for 'q' key to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
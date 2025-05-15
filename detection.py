import cv2
import numpy as np
from ultralytics import YOLO
import json

# Global variables for dynamic color thresholds
min_S = 100.0
min_V = 100.0

# Load calibration if exists
try:
    with open('calibration.json', 'r') as f:
        data = json.load(f)
        min_S = data['min_S']
        min_V = data['min_V']
    print(f"Loaded calibration: min_S={min_S}, min_V={min_V}")
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    min_S = 100.0
    min_V = 100.0
    print("Using default calibration: min_S=100.0, min_V=100.0")

def detect_red_bricks(frame, model):
    """Detect red bricks in the frame using YOLO and dynamic HSV color filtering."""
    global min_S, min_V
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
            
            # Define red color ranges with dynamic S and V thresholds
            lower_red1 = np.array([0, int(min_S), int(min_V)])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, int(min_S), int(min_V)])
            upper_red2 = np.array([179, 255, 255])
            
            # Create masks for red color
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            #cv2.imshow('HSV Mask', mask)  # Debug

            # Calculate percentage of red pixels
            red_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Classify as red brick if more than 50% of pixels are red
            if total_pixels > 0 and red_pixels / total_pixels > 0.2:
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, 'Red Brick', (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image

def calibrate_color(frame):
    """Calibrate the red brick color by selecting a region and adjusting HSV thresholds."""
    global min_S, min_V
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_x, start_y, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (start_x, start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow('Calibrate Color', temp_frame)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_x, end_y = x, y
            x1 = min(start_x, end_x)
            x2 = max(start_x, end_x)
            y1 = min(start_y, end_y)
            y2 = max(start_y, end_y)
            if x2 > x1 and y2 > y1:
               V2 = cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               roi = frame[y1:y2, x1:x2]
               hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
               S_channel = hsv_roi[:, :, 1]
               V_channel = hsv_roi[:, :, 2]
               mean_S, std_S = np.mean(S_channel), np.std(S_channel)
               mean_V, std_V = np.mean(V_channel), np.std(V_channel)
               min_S = max(0, mean_S - 2 * std_S)
               min_V = max(0, mean_V - 2 * std_V)
               print(f"Calibrated: min_S = {min_S:.1f}, min_V = {min_V:.1f}")
               # Save calibration
               with open('calibration.json', 'w') as f:
                   json.dump({'min_S': min_S, 'min_V': min_V}, f)
               print("Saved calibration to file")
    
    drawing = False
    start_x, start_y = -1, -1
    temp_frame = frame.copy()
    cv2.namedWindow('Calibrate Color')
    cv2.setMouseCallback('Calibrate Color', mouse_callback)
    cv2.imshow('Calibrate Color', temp_frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
    cv2.destroyWindow('Calibrate Color')

# Load YOLO model (ensure the model file is available)
model = YOLO('yolo11s.pt')

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit, 'c' to calibrate color")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Detect red bricks
    output_image = detect_red_bricks(frame, model)
    
    # Add instruction text
    cv2.putText(output_image, "Press 'c' to calibrate color", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    # Add current thresholds text
    cv2.putText(output_image, f"Thresholds: min_S={min_S:.1f}, min_V={min_V:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1)
    
    # Display the output
    cv2.imshow('Red Brick Detection', output_image)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibrate_color(frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
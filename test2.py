import cv2
import numpy as np

def detect_red_boxes(frame):
    """
    Detect red boxes in the frame and return their contours and corners
    """
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    # Red has two ranges in HSV due to wraparound
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_boxes = []
    
    for contour in contours:
        # Filter contours by area to avoid noise
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the approximated contour has 4 vertices (rectangle-like)
            if len(approx) >= 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio to filter out non-box shapes
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio for boxes
                    red_boxes.append({
                        'contour': contour,
                        'approx': approx,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
    
    return red_boxes, mask

def draw_corners_and_info(frame, red_boxes):
    """
    Draw yellow points at corners of the green contour outline
    """
    for i, box in enumerate(red_boxes):
        contour = box['contour']
        approx = box['approx']
        x, y, w, h = box['bbox']
        center = box['center']
        area = box['area']
        
        # Draw the contour outline in green
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Draw yellow circles at the corners of the approximated contour polygon
        for point in approx:
            cv2.circle(frame, tuple(point[0]), 8, (0, 255, 255), -1)
        
        # Add text information showing position
        info_text = f"Box {i+1}: ({center[0]}, {center[1]})"
        cv2.putText(frame, info_text, (center[0]-50, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        
        
        # Draw center point in red
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Add text information
        info_text = f"Box {i+1}: ({center[0]}, {center[1]})"
        cv2.putText(frame, info_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Red Box Detection Started!")
    print("Instructions:")
    print("- Show red rectangular objects to the camera")
    print("- Yellow points will mark the corners")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect red boxes
        red_boxes, mask = detect_red_boxes(frame)
        
        # Draw corners and information
        draw_corners_and_info(frame, red_boxes)
        
        # Add status information
        status_text = f"Red boxes detected: {len(red_boxes)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frames
        cv2.imshow('Red Box Detection', frame)
        cv2.imshow('Red Mask', mask)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"red_box_detection_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
            frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Red box detection ended.")

if __name__ == "__main__":
    main()
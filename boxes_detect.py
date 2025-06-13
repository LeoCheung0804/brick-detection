import cv2
import numpy as np

class RedBoxDetector:
    def __init__(self):
        # Target color RGB(170, 74, 68) converted to HSV for better color detection
        # We'll define a range around this color
        target_rgb = np.array([[[170, 74, 68]]], dtype=np.uint8)
        # target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0]  # computed but not used
        
        # Define HSV range for red color detection
        # Red can wrap around in HSV, so we need two ranges
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Minimum area for detected boxes (adjust based on expected box size)
        self.min_area = 500
        self.max_area = 50000
        
        # Calibration variables
        self.calibration_mode = False
        self.roi_start = None
        self.roi_end = None
        self.selecting_roi = False
        
    def mouse_callback(self, event, x, y, _flags, _param):
        """Mouse callback for ROI selection during calibration"""
        if not self.calibration_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.selecting_roi = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting_roi:
            self.roi_end = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
            self.selecting_roi = False
    
    def calibrate_color(self, frame):
        """Calibrate HSV thresholds based on selected ROI"""
        if self.roi_start is None or self.roi_end is None:
            return False

        # Get ROI coordinates
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end

        # Ensure proper order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        # Convert ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculate mean and std for each HSV channel
        h_mean = np.mean(hsv_roi[:, :, 0])
        s_mean = np.mean(hsv_roi[:, :, 1])
        v_mean = np.mean(hsv_roi[:, :, 2])

        h_std = np.std(hsv_roi[:, :, 0])
        s_std = np.std(hsv_roi[:, :, 1])
        v_std = np.std(hsv_roi[:, :, 2])

        # Create new thresholds based on mean ± 2*std
        h_tolerance = max(10, 2 * h_std)
        s_tolerance = max(30, 2 * s_std)
        v_tolerance = max(30, 2 * v_std)

        # Handle red color wrap-around in HSV
        if h_mean < 30 or h_mean > 150:  # Likely red color
            if h_mean < 30:
                self.lower_red1 = np.array([
                    max(0, int(h_mean - h_tolerance)),
                    max(30, int(s_mean - s_tolerance)),
                    max(30, int(v_mean - v_tolerance))
                ], dtype=np.uint8)
                self.upper_red1 = np.array([
                    min(30, int(h_mean + h_tolerance)),
                    255,
                    255
                ], dtype=np.uint8)

                self.lower_red2 = np.array([
                    max(150, int(180 - h_tolerance)),
                    max(30, int(s_mean - s_tolerance)),
                    max(30, int(v_mean - v_tolerance))
                ], dtype=np.uint8)
                self.upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
            else:
                self.lower_red1 = np.array([
                    0,
                    max(30, int(s_mean - s_tolerance)),
                    max(30, int(v_mean - v_tolerance))
                ], dtype=np.uint8)
                self.upper_red1 = np.array([
                    min(30, int(h_tolerance)),
                    255,
                    255
                ], dtype=np.uint8)

                self.lower_red2 = np.array([
                    max(150, int(h_mean - h_tolerance)),
                    max(30, int(s_mean - s_tolerance)),
                    max(30, int(v_mean - v_tolerance))
                ], dtype=np.uint8)
                self.upper_red2 = np.array([
                    min(180, int(h_mean + h_tolerance)),
                    255,
                    255
                ], dtype=np.uint8)
        else:
            self.lower_red1 = np.array([
                max(0, int(h_mean - h_tolerance)),
                max(30, int(s_mean - s_tolerance)),
                max(30, int(v_mean - v_tolerance))
            ], dtype=np.uint8)
            self.upper_red1 = np.array([
                min(180, int(h_mean + h_tolerance)),
                255,
                255
            ], dtype=np.uint8)
            self.lower_red2 = np.array([180, 0, 0], dtype=np.uint8)
            self.upper_red2 = np.array([180, 0, 0], dtype=np.uint8)

        print(f"\nColor Calibration Complete!")
        print(f"HSV Mean: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
        print(f"Range 1: {self.lower_red1} to {self.upper_red1}")
        print(f"Range 2: {self.lower_red2} to {self.upper_red2}")

        return True
    
    def draw_calibration_interface(self, frame):
        """Draw calibration interface elements"""
        if not self.calibration_mode:
            return frame
            
        # Draw instructions
        instructions = [
            "CALIBRATION MODE",
            "1. Click and drag to select box region",
            "2. Press SPACE to apply calibration",
            "3. Press ESC to exit calibration",
            "4. Press 'r' to reset selection"
        ]
        
        for i, text in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        # Draw ROI rectangle if selecting
        if self.roi_start and self.roi_end:
            cv2.rectangle(frame, self.roi_start, self.roi_end, (0, 255, 255), 2)
            
        return frame
    
    def create_red_mask(self, frame):
        """Create a mask for red objects in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_boxes(self, frame):
        """Detect red boxes and return their properties"""
        mask = self.create_red_mask(frame)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area < area < self.max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the shape is roughly rectangular (aspect ratio)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Adjust this range based on your boxes
                    
                    # Calculate center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Define corners
                    corners = {
                        'top_left': (x, y),
                        'top_right': (x + w, y),
                        'bottom_left': (x, y + h),
                        'bottom_right': (x + w, y + h)
                    }
                    
                    # Store box information
                    box_info = {
                        'center': (center_x, center_y),
                        'corners': corners,
                        'bounding_rect': (x, y, w, h),
                        'area': area
                    }
                    
                    detected_boxes.append(box_info)
        
        return detected_boxes
    
    def draw_detections(self, frame, boxes):
        """Draw detection results on the frame"""
        for i, box in enumerate(boxes):
            x, y, w, h = box['bounding_rect']
            center = box['center']
            corners = box['corners']
            
            # Draw bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            
            # Draw corners
            for corner_pos in corners.values():
                cv2.circle(frame, corner_pos, 3, (0, 0, 255), -1)
            
            # Add labels
            cv2.putText(frame, f'Box {i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Center: {center}', (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

def main():
    # Initialize the detector
    detector = RedBoxDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set up mouse callback
    cv2.namedWindow('Red Box Detection')
    cv2.setMouseCallback('Red Box Detection', detector.mouse_callback)
    
    print("Red Box Detection Started")
    print("Controls:")
    print("  'q' - quit")
    print("  's' - save current frame")
    print("  'c' - enter calibration mode")
    print("  'SPACE' - apply calibration (in calibration mode)")
    print("  'ESC' - exit calibration mode")
    print("  'r' - reset ROI selection (in calibration mode)")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Handle calibration mode
        if detector.calibration_mode:
            frame_with_interface = detector.draw_calibration_interface(frame.copy())
            cv2.imshow('Red Box Detection', frame_with_interface)
        else:
            # Normal detection mode
            detected_boxes = detector.detect_boxes(frame)
            frame_with_detections = detector.draw_detections(frame, detected_boxes)
            
            # Display information
            info_text = f"Boxes detected: {len(detected_boxes)}"
            cv2.putText(frame_with_detections, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add calibration hint
            cv2.putText(frame_with_detections, "Press 'c' for calibration", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Print detection info to console
            if detected_boxes:
                print(f"\nFrame {frame_count}: {len(detected_boxes)} boxes detected")
                for i, box in enumerate(detected_boxes):
                    print(f"  Box {i+1}:")
                    print(f"    Center: {box['center']}")
                    print(f"    Corners: {box['corners']}")
                    print(f"    Area: {box['area']}")
            
            cv2.imshow('Red Box Detection', frame_with_detections)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not detector.calibration_mode:
            filename = f'detection_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame_with_detections)
            print(f"Frame saved as {filename}")
        elif key == ord('c') and not detector.calibration_mode:
            print("\nEntering calibration mode...")
            detector.calibration_mode = True
            detector.roi_start = None
            detector.roi_end = None
        elif key == 27:  # ESC key
            if detector.calibration_mode:
                print("Exiting calibration mode")
                detector.calibration_mode = False
                detector.roi_start = None
                detector.roi_end = None
        elif key == ord(' ') and detector.calibration_mode:  # SPACE key
            if detector.calibrate_color(frame):
                print("Calibration applied successfully!")
                detector.calibration_mode = False
                detector.roi_start = None
                detector.roi_end = None
            else:
                print("Please select a valid region first")
        elif key == ord('r') and detector.calibration_mode:  # Reset ROI
            detector.roi_start = None
            detector.roi_end = None
            print("ROI selection reset")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main()
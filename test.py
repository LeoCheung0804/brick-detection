import cv2
import numpy as np
from ultralytics import YOLO


# Global variables for dynamic color thresholds
lower_red1 = np.array([0, 43, 46])
upper_red1 = np.array([25, 255, 255])
lower_red2 = np.array([156, 43, 46])
upper_red2 = np.array([180, 255, 255])
        

def detect_boxes():
    # Load the YOLO model (adjust the model path as needed)
    model = YOLO("yolo11n.pt")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break


        # Convert frame to HSV and apply the dynamic color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        filtered = cv2.bitwise_and(frame, frame, mask=mask)

        # Run detection on the original frame using YOLO
        results = model(filtered, verbose=False)

        # Draw the detected boxes on the frame
        for result in results:
            if result.boxes is not None:
                for box in result.boxes.xyxy:  # xyxy format: x1, y1, x2, y2
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the original frame with boxes and the dynamically filtered frame
        cv2.imshow("Detected Boxes", frame)
        cv2.imshow("Dynamic HSV Filter", filtered)

        key = cv2.waitKey(1) & 0xFF
        # Exit on 'q' key
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_boxes()


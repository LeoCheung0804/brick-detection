import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Brick detection using YOLOv8 and OpenCV')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='YOLOv8 model path or name (default: yolov8n.pt)')
    parser.add_argument('--classes', type=str, default='brick', 
                        help='Class names to detect, comma separated (default: brick)')
    parser.add_argument('--show-fps', action='store_true', help='Display FPS counter')
    return parser.parse_args()

def main():
    """Main function for brick detection"""
    # Parse arguments
    args = parse_arguments()
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Get class names of interest
    class_names = args.classes.split(',')
    class_ids = []
    
    # Map class names to model's class IDs
    model_classes = model.names
    for name in class_names:
        name = name.strip().lower()
        # Find matching class in model
        for id, model_name in model_classes.items():
            if name in model_name.lower():
                class_ids.append(id)
                print(f"Will detect class: {model_name} (ID: {id})")
    
    # If no matching classes found, detect everything
    if not class_ids:
        print("No matching classes found. Will detect all objects.")
    
    # Initialize webcam
    print(f"Opening camera {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("Starting detection. Press 'q' to quit.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to receive frame from camera")
            break
        
        # Perform YOLO detection
        results = model(frame, conf=args.conf)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box class
                cls_id = int(box.cls.item())
                
                # Skip if we're only looking for specific classes and this isn't one
                if class_ids and cls_id not in class_ids:
                    continue
                
                # Get bounding box coordinates (convert to integers)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get confidence score
                conf = float(box.conf.item())
                
                # Get class name
                cls_name = model.names[cls_id]
                
                # Draw bounding box and label
                color = (0, 255, 0)  # Green for bricks
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name and confidence
                label = f"{cls_name}: {conf:.2f}"
                
                # Calculate text size for proper background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Draw text background
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                
                # Draw text label
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Calculate and display FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        if args.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Brick Detection", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
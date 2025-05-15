# Red Brick Detection with YOLO and Dynamic HSV Calibration

This project provides a Python script (`detection.py`) for real-time detection of red LEGO bricks using a webcam. It utilizes the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) object detection model and dynamic HSV color threshold calibration for robust red color filtering.

## Features

- **YOLO-based object detection** to identify brick-like objects.
- **Dynamic HSV color calibration**: Easily adapt detection to your lighting by selecting a region of a red brick.
- **Real-time webcam processing** with OpenCV.
- **Persistent calibration**: HSV thresholds are saved and loaded automatically.

## Requirements

- Python 3.7+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [Ultralytics](https://pypi.org/project/ultralytics/) (for YOLO)
- A YOLO model file (default: `yolo11s.pt` in the project directory)
- Webcam

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Place your YOLO model file (e.g., `yolo11s.pt`) in the project directory.
2. Run the script:

    ```sh
    python detection.py
    ```

### Controls

- **q**: Quit the program.
- **c**: Calibrate color. Draw a rectangle around a red brick in the calibration window to set optimal HSV thresholds.

### Calibration

- Press **c** to open the calibration window.
- Draw a rectangle around a red brick using your mouse.
- The script will calculate and save new HSV thresholds for improved detection under your lighting conditions.

### Detection

- Detected red bricks are highlighted with a green rectangle and labeled "Red Brick".
- Current HSV thresholds are displayed on the video feed.

## Files

- `detection.py`: Main detection and calibration script.
- `calibration.json`: (Auto-generated) Stores the last used HSV thresholds.

## Notes

- Ensure your webcam is connected and accessible.
- The script uses the first available camera (`cv2.VideoCapture(0)`).
- You can adjust the YOLO model path in the script if needed.

## License

This project is for educational and research purposes.

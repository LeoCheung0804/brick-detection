import cv2
import numpy as np

def calibrate_camera_online(chessboard_size, frame_size, capture_device_index=0, output_file='calibration_data.npz'):
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the chessboard size
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Open video capture
    cap = cv2.VideoCapture(capture_device_index)

    while len(objpoints) < 10:  # Collect 10 valid frames for calibration
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('gray', gray)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
            cv2.imshow('Calibration', frame)
            
            if cv2.waitKey(0) & 0xFF == ord('a'):
                # Add the object points and image points to the lists   
                objpoints.append(objp)
                imgpoints.append(corners2)

        cv2.imshow('Calibration', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    # Save the results to a file
    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    return camera_matrix, dist_coeffs

# Example usage
chessboard_size = (6, 5)
frame_size = (640, 480)
camera_matrix, dist_coeffs = calibrate_camera_online(chessboard_size, frame_size)
print(camera_matrix)
print(dist_coeffs)

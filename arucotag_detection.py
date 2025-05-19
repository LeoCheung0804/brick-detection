import cv2
import numpy as np
import os
from collections import deque
import json
import socket
import threading
from typing import Optional, Dict

# Add these as global variables at the top of the file, after imports
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

offset_x = 0.01
offset_y = -0.02
offset_z = 0.02
robot_home_pos = {
        "x": 7.3+offset_x, 
        "y": 6.2+offset_y, 
        "z": -2.7+offset_z, 
        "roll": 0, 
        "pitch": 0, 
        "yaw": -0.0344
    }


def load_camera_data(file_path):
    # Load camera matrix and distortion coefficients from a file
    with np.load(file_path) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    return camera_matrix, dist_coeffs

def apply_low_pass_filter(new_value, history):
    history.append(new_value)
    result = np.mean(history, axis=0)
    # print("result: ", result)
    return result

def detect_aruco_markers(image, camera_matrix, dist_coeffs):
    global home_position, home_rotation_matrix, camera_position, camera_rotation_matrix
    global position_history, rotation_history

    global real_position, real_rotation, distance
    

    # Load the dictionary that was used to generate the markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Create ArUco board
    board = cv2.aruco.GridBoard(
        (3, 4),
        0.05,
        0.01,
        aruco_dict
    )

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if ids is not None:
        # Clear console before new output
        os.system('cls' if os.name == 'nt' else 'clear')
        # Estimate the board pose
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board, camera_matrix, dist_coeffs, None, None
        )

        if retval:
            # Draw board axis
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
            
            # Convert board rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Calculate camera position (inverse transform)
            camera_position = -np.matrix(rmat).T * np.matrix(tvec)
            
            # Camera rotation is inverse of board rotation
            camera_rotation_matrix = rmat.T
            # camera_rotation_vec, _ = cv2.Rodrigues(camera_rotation_matrix)
            
            # Apply low-pass filter to position and rotation
            camera_position = apply_low_pass_filter(np.array(camera_position), position_history)
            camera_rotation_matrix = apply_low_pass_filter(camera_rotation_matrix, rotation_history)
            
            # Ensure rotation matrix stays orthogonal after filtering
            u, _, vt = np.linalg.svd(camera_rotation_matrix)
            camera_rotation_matrix = u @ vt
            
            # Convert to matrix format for consistency with rest of code
            camera_position = np.matrix(camera_position).T
            camera_rotation_matrix = np.matrix(camera_rotation_matrix)
            
            # Draw home position axes if set
            # if home_position is not None:
            #     # Calculate relative position and check distance
            #     relative_position = home_position - camera_position
            #     distance = np.linalg.norm(relative_position)
                
            #     # Only draw if distance is greater than threshold (e.g., 0.1 meters)
            #     if distance > 0.1:
            #         # Calculate the transformation from current camera to home camera
            #         relative_rotation = home_rotation_matrix @ camera_rotation_matrix.T
            #         relative_position = camera_rotation_matrix @ relative_position
                    
            #         # Convert rotation to rotation vector
            #         relative_rvec, _ = cv2.Rodrigues(relative_rotation)
                    
            #         # Draw home camera frame from current camera's perspective
            #         cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, relative_rvec, relative_position, 0.15)

            # Convert rotation matrix to Euler angles (in radians)
            # Order of rotations: yaw (Y), pitch (X), roll (Z)
            pitch = np.arctan2(-camera_rotation_matrix[2,0], 
                             np.sqrt(camera_rotation_matrix[2,1]**2 + camera_rotation_matrix[2,2]**2))
            yaw = np.arctan2(camera_rotation_matrix[2,1], camera_rotation_matrix[2,2])
            roll = np.arctan2(camera_rotation_matrix[1,0], camera_rotation_matrix[0,0])
            pitch = pitch
            yaw = yaw
            roll = roll

            # Convert to degrees
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)
            
            # Store current position and rotation for relative calculations
            current_position = camera_position
            current_rotation_matrix = camera_rotation_matrix
            
            # Calculate relative position and rotation if home is set
            if home_position is not None:
                relative_position = current_position - home_position
                relative_position[0,0] = relative_position[0,0]
                relative_position[0,1] = -relative_position[0,1]
                relative_position[0,2] = -relative_position[0,2]
                relative_rotation = current_rotation_matrix @ home_rotation_matrix.T
                
                distance = np.linalg.norm(relative_position)
                # Convert relative rotation to euler angles
                pitch = np.arctan2(-relative_rotation[2,0], 
                                 np.sqrt(relative_rotation[2,1]**2 + relative_rotation[2,2]**2))
                yaw = np.arctan2(relative_rotation[2,1], relative_rotation[2,2])
                roll = np.arctan2(relative_rotation[1,0], relative_rotation[0,0])
                
                # Convert to degrees
                rel_yaw_deg = np.degrees(yaw) # roll
                rel_pitch_deg = -np.degrees(pitch) # pitch
                rel_roll_deg = -np.degrees(roll) # yaw
                
                real_position = [
                    relative_position[0,0] + robot_home_pos['x'],
                    relative_position[0,1] + robot_home_pos['y'],
                    relative_position[0,2] + robot_home_pos['z']
                ]
                real_rotation = [
                    np.radians(rel_yaw_deg + robot_home_pos['roll']),
                    np.radians(rel_pitch_deg + robot_home_pos['pitch']),
                    np.radians(rel_roll_deg + robot_home_pos['yaw'])
                ]

                print(f"Current camera position (m):")
                print(f"  X: {camera_position[0,0]:.3f}")
                print(f"  Y: {camera_position[0,1]:.3f}")
                print(f"  Z: {camera_position[0,2]:.3f}")
                print(f"Current camera angles (degrees):")
                print(f"  Yaw (Y-axis): {yaw_deg:.2f}")
                print(f"  Pitch (X-axis): {pitch_deg:.2f}")
                print(f"  Roll (Z-axis): {roll_deg:.2f}")
                print("\nRelative to home:")
                print(f"Position (m):")
                print(f"  X: {relative_position[0,0]:.3f}")
                print(f"  Y: {relative_position[0,1]:.3f}")
                print(f"  Z: {relative_position[0,2]:.3f}")
                print(f"Angles (degrees):")
                # print(f"  Yaw: {rel_yaw_deg:.2f}")
                # print(f"  Pitch: {rel_pitch_deg:.2f}")
                # print(f"  Roll: {rel_roll_deg:.2f}")
                print(f"  Roll (1): {rel_yaw_deg:.2f}")
                print(f"  Pitch (2): {rel_pitch_deg:.2f}")
                print(f"  Yaw (3): {rel_roll_deg:.2f}")
                print(f"Real Position(m):")
                print(f"  X: {real_position[0]:.3f}")
                print(f"  Y: {real_position[1]:.3f}")
                print(f"  Z: {real_position[2]:.3f}")
                print(f"  Roll (1): {real_rotation[0]:.3f}")
                print(f"  Pitch (2): {real_rotation[1]:.3f}")
                print(f"  Yaw (3): {real_position[2]:.3f}")
                print(f"Distance: {distance:.3f} meters")
                print(f"")
            else:
                print(f"Camera position (m):")
                print(f"  X: {camera_position[0,0]:.3f}")
                print(f"  Y: {camera_position[0,1]:.3f}")
                print(f"  Z: {camera_position[0,2]:.3f}")
                print(f"Camera angles (degrees):")
                print(f"  Yaw (Y-axis): {yaw_deg:.2f}")
                print(f"  Pitch (X-axis): {pitch_deg:.2f}")
                print(f"  Roll (Z-axis): {roll_deg:.2f}")
                print("\nPress 'h' to set home position")
    else:
        print("No ArUco markers detected")

    return image

def save_home_position(file_path='home_position.json'):
    """Save home position and rotation matrix to a JSON file"""
    if home_position is not None and home_rotation_matrix is not None:
        data = {
            'position': home_position.tolist(),
            'rotation': home_rotation_matrix.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f"\nHome position saved to {file_path}")

def load_home_position(file_path='home_position.json'):
    """Load home position and rotation matrix from a JSON file"""
    global home_position, home_rotation_matrix
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            home_position = np.matrix(data['position'])
            home_rotation_matrix = np.matrix(data['rotation'])
        print(f"\nHome position loaded from {file_path}")
        return True
    except FileNotFoundError:
        print(f"\nNo saved home position found at {file_path}")
        return False

def get_current_position() -> Optional[Dict]:
    """Return the current position and rotation data if available"""
    if real_position is None or real_rotation is None:
        return None

    return {
        "x": float(real_position[0]),
        "y": float(real_position[1]),
        "z": float(real_position[2]),
        "roll": float(real_rotation[0]),
        "pitch": float(real_rotation[1]),
        "yaw": float(real_rotation[2]),
        "distance": float(distance)
    }

def run_udp_server():
    """Run UDP server to handle position requests"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")
    
    while True:
        data, addr = sock.recvfrom(1024)
        if data.decode() == "get_position":
            position_data = get_current_position()
            if position_data:
                response = json.dumps(position_data)
                sock.sendto(response.encode(), addr)
            else:
                sock.sendto(b"No position data available", addr)

# Modify the main execution block at the bottom of the file
if __name__ == "__main__":
    # Add new globals for UDP server
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5000
    last_position: Optional[Dict] = None
    # Start UDP server in a separate thread
    udp_thread = threading.Thread(target=run_udp_server, daemon=True)
    udp_thread.start()
    
    # Load camera data and home position
    camera_matrix, dist_coeffs = load_camera_data('calibration_data_1080.npz')
    load_home_position()
    
    # open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_image = detect_aruco_markers(frame, camera_matrix, dist_coeffs)
        cv2.imshow('Detected ArUco markers', output_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('h'):
            if 'camera_position' in locals() and 'camera_rotation_matrix' in locals():
                home_position = camera_position
                home_rotation_matrix = camera_rotation_matrix
                save_home_position()
                print("\nHome position set!")
        elif key == ord('l'):
            load_home_position()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
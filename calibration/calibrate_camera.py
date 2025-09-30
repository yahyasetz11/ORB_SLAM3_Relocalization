#!/usr/bin/env python3
"""
Camera Calibration Script for ORB_SLAM3
Generates camera parameters from checkerboard calibration video

Run :
python3 calibrate_camera.py path/to/your/calibration_video.mp4
"""

import cv2
import numpy as np
import sys
import yaml

def calibrate_camera_from_video(video_path, checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate camera from video containing checkerboard patterns
    
    Args:
        video_path: Path to calibration video
        checkerboard_size: (columns-1, rows-1) of internal corners
        square_size: Size of checkerboard square in meters
    """
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height} @ {fps} fps")
    print(f"Looking for {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard pattern...")
    
    frame_count = 0
    used_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            used_frames += 1
            
            # Draw and display corners
            cv2.drawChessboardCorners(frame, checkerboard_size, corners2, ret)
            cv2.putText(frame, f"Frame {used_frames} captured", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frames: {used_frames}/{frame_count}", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Calibration', frame)
        
        # Skip frames for faster processing
        if frame_count % 10 != 0:
            continue
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if used_frames < 10:
        print(f"Error: Not enough frames with checkerboard pattern found ({used_frames}/10 minimum)")
        return None
    
    print(f"\nCalibrating with {used_frames} frames...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                        (width, height), None, None)
    
    if not ret:
        print("Calibration failed!")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    avg_error = total_error / len(objpoints)
    
    # Extract parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    
    k1, k2, p1, p2, k3 = dist[0]
    
    print("\n=== Calibration Results ===")
    print(f"Reprojection error: {avg_error:.4f} pixels")
    print(f"fx: {fx:.2f}")
    print(f"fy: {fy:.2f}")
    print(f"cx: {cx:.2f}")
    print(f"cy: {cy:.2f}")
    print(f"k1: {k1:.6f}")
    print(f"k2: {k2:.6f}")
    print(f"p1: {p1:.6f}")
    print(f"p2: {p2:.6f}")
    print(f"k3: {k3:.6f}")
    
    # Create ORB_SLAM3 compatible YAML
    yaml_content = {
        '%YAML': 1.0,
        'Camera.fx': float(fx),
        'Camera.fy': float(fy),
        'Camera.cx': float(cx),
        'Camera.cy': float(cy),
        'Camera.k1': float(k1),
        'Camera.k2': float(k2),
        'Camera.p1': float(p1),
        'Camera.p2': float(p2),
        'Camera.k3': float(k3),
        'Camera.width': int(width),
        'Camera.height': int(height),
        'Camera.fps': float(fps),
        'Camera.RGB': 1,
        'ORBextractor.nFeatures': 1000,
        'ORBextractor.scaleFactor': 1.2,
        'ORBextractor.nLevels': 8,
        'ORBextractor.iniThFAST': 20,
        'ORBextractor.minThFAST': 7,
        'Viewer.KeyFrameSize': 0.05,
        'Viewer.KeyFrameLineWidth': 1,
        'Viewer.GraphLineWidth': 0.9,
        'Viewer.PointSize': 2,
        'Viewer.CameraSize': 0.08,
        'Viewer.CameraLineWidth': 3,
        'Viewer.ViewpointX': 0,
        'Viewer.ViewpointY': -0.7,
        'Viewer.ViewpointZ': -1.8,
        'Viewer.ViewpointF': 500
    }
    
    # Save to YAML file
    output_file = 'camera_calibration.yaml'
    with open(output_file, 'w') as f:
        f.write('%YAML:1.0\n\n')
        f.write('#--------------------------------------------------------------------------------------------\n')
        f.write('# Camera Parameters (Calibrated)\n')
        f.write('#--------------------------------------------------------------------------------------------\n\n')
        
        for key, value in yaml_content.items():
            if key != '%YAML':
                if isinstance(value, float):
                    f.write(f'{key}: {value:.6f}\n')
                else:
                    f.write(f'{key}: {value}\n')
                    
                if key == 'Camera.RGB':
                    f.write('\n#--------------------------------------------------------------------------------------------\n')
                    f.write('# ORB Parameters\n')
                    f.write('#--------------------------------------------------------------------------------------------\n')
                elif key == 'ORBextractor.minThFAST':
                    f.write('\n#--------------------------------------------------------------------------------------------\n')
                    f.write('# Viewer Parameters\n')
                    f.write('#--------------------------------------------------------------------------------------------\n')
    
    print(f"\nCalibration saved to: {output_file}")
    print(f"Use this file as settings for ORB_SLAM3")
    
    return mtx, dist


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate_camera.py <video_path> [checkerboard_cols] [checkerboard_rows]")
        print("Example: python calibrate_camera.py calibration.mp4 9 6")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Default checkerboard size (9x6 internal corners = 10x7 squares)
    cols = 9 if len(sys.argv) < 3 else int(sys.argv[2])
    rows = 6 if len(sys.argv) < 4 else int(sys.argv[3])
    
    calibrate_camera_from_video(video_path, (cols, rows))
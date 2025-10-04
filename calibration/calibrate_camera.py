#!/usr/bin/env python3
"""
Camera Calibration Script for ORB_SLAM3 - HEADLESS VERSION
No display window, faster processing
"""

import cv2
import numpy as np
import sys
import time

def calibrate_camera_from_video(video_path, checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate camera WITHOUT display window
    """
    
    print("="*100)
    print("CALIBRATION HEADLESS MODE - No Display")
    print("="*100)
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []
    
    # Open video
    print(f"\n📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"\n🔍 Looking for {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard pattern...")
    print("   (This may take a minute...)\n")
    
    frame_count = 0
    used_frames = 0
    start_time = time.time()
    last_update = time.time()
    
    # Process every Nth frame (adjust for speed)
    skip_frames = 5  # Process every 5th frame (faster)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing
        if frame_count % skip_frames != 0:
            continue
        
        # Progress update every 2 seconds
        if time.time() - last_update > 2:
            progress = (frame_count / total_frames) * 100
            print(f"   Processing: {progress:.1f}% - Found {used_frames} calibration frames", end='\r')
            last_update = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            used_frames += 1
            
            # Limit number of frames used (for speed)
            if used_frames >= 100:  # Use max 100 frames
                print(f"\n✅ Collected enough frames ({used_frames})")
                break
    
    cap.release()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Frame extraction took: {elapsed:.1f} seconds")
    
    if used_frames < 10:
        print(f"\n❌ Error: Not enough frames with checkerboard pattern found ({used_frames}/10 minimum)")
        print("\n💡 Tips:")
        print("   - Check if checkerboard size is correct (currently {0}x{1})".format(*checkerboard_size))
        print("   - Ensure good lighting in video")
        print("   - Pattern should be clearly visible")
        return None
    
    print(f"\n🔧 Calibrating with {used_frames} frames...")
    print("   (This may take a moment...)")
    
    # Calibrate camera
    calib_start = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, 
        (width, height), None, None
    )
    calib_time = time.time() - calib_start
    
    if not ret:
        print("❌ Calibration failed!")
        return None
    
    print(f"✅ Calibration completed in {calib_time:.1f} seconds")
    
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
    
    print("\n" + "="*100)
    print("📊 CALIBRATION RESULTS")
    print("="*100)
    print(f"Reprojection error: {avg_error:.4f} pixels", end="")
    if avg_error < 0.5:
        print(" ✅ Excellent!")
    elif avg_error < 1.0:
        print(" ✅ Good")
    else:
        print(" ⚠️ High - consider recalibrating")
    
    print(f"\nCamera parameters:")
    print(f"  fx: {fx:.2f}")
    print(f"  fy: {fy:.2f}")
    print(f"  cx: {cx:.2f}")
    print(f"  cy: {cy:.2f}")
    print(f"\nDistortion coefficients:")
    print(f"  k1: {k1:.6f}")
    print(f"  k2: {k2:.6f}")
    print(f"  p1: {p1:.6f}")
    print(f"  p2: {p2:.6f}")
    print(f"  k3: {k3:.6f}")
    
    # Save to YAML file
    output_file = 'camera_calibration.yaml'
    with open(output_file, 'w') as f:
        f.write('%YAML:1.0\n\n')
        f.write('#--------------------------------------------------------------------------------------------\n')
        f.write('# Camera Parameters (Calibrated)\n')
        f.write('#--------------------------------------------------------------------------------------------\n\n')
        
        f.write(f'Camera.fx: {fx:.6f}\n')
        f.write(f'Camera.fy: {fy:.6f}\n')
        f.write(f'Camera.cx: {cx:.6f}\n')
        f.write(f'Camera.cy: {cy:.6f}\n')
        f.write(f'\nCamera.k1: {k1:.6f}\n')
        f.write(f'Camera.k2: {k2:.6f}\n')
        f.write(f'Camera.p1: {p1:.6f}\n')
        f.write(f'Camera.p2: {p2:.6f}\n')
        f.write(f'Camera.k3: {k3:.6f}\n')
        f.write(f'\nCamera.width: {width}\n')
        f.write(f'Camera.height: {height}\n')
        f.write(f'Camera.fps: {fps:.1f}\n')
        f.write(f'Camera.RGB: 1\n')
        
        f.write('\n#--------------------------------------------------------------------------------------------\n')
        f.write('# ORB Parameters\n')
        f.write('#--------------------------------------------------------------------------------------------\n')
        f.write('ORBextractor.nFeatures: 1000\n')
        f.write('ORBextractor.scaleFactor: 1.2\n')
        f.write('ORBextractor.nLevels: 8\n')
        f.write('ORBextractor.iniThFAST: 20\n')
        f.write('ORBextractor.minThFAST: 7\n')
        
        f.write('\n#--------------------------------------------------------------------------------------------\n')
        f.write('# Viewer Parameters\n')
        f.write('#--------------------------------------------------------------------------------------------\n')
        f.write('Viewer.KeyFrameSize: 0.05\n')
        f.write('Viewer.KeyFrameLineWidth: 1\n')
        f.write('Viewer.GraphLineWidth: 0.9\n')
        f.write('Viewer.PointSize: 2\n')
        f.write('Viewer.CameraSize: 0.08\n')
        f.write('Viewer.CameraLineWidth: 3\n')
        f.write('Viewer.ViewpointX: 0\n')
        f.write('Viewer.ViewpointY: -0.7\n')
        f.write('Viewer.ViewpointZ: -1.8\n')
        f.write('Viewer.ViewpointF: 1000\n')
    
    print(f"\n💾 Calibration saved to: {output_file}")
    print("✅ Use this file as settings for ORB_SLAM3")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total processing time: {total_time:.1f} seconds")
    
    return mtx, dist

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate_camera_headless.py <video_path> [checkerboard_cols] [checkerboard_rows]")
        print("Example: python calibrate_camera_headless.py calibration.mp4 9 6")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Default checkerboard size
    cols = 9 if len(sys.argv) < 3 else int(sys.argv[2])
    rows = 6 if len(sys.argv) < 4 else int(sys.argv[3])
    
    calibrate_camera_from_video(video_path, (cols, rows))
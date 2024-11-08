import cv2
import numpy as np
import time

def get_camera_properties(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    # Get camera name (this might be OS and driver-dependent)
    camera_name = cap.get(cv2.CAP_PROP_BACKEND)
    
    # Get frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    return camera_name, fps

def check_camera_source(camera_index):
    # Check camera properties
    camera_properties = get_camera_properties(camera_index)
    if not camera_properties:
        return "Error: Cannot access camera."

    camera_name, fps = camera_properties
    if camera_name is None:
        return "Error: Cannot retrieve camera name."
    
    # Check frame rate
    if fps not in [30, 60]:
        print(fps)
        return "Virtual Camera Detected (Unusual frame rate)."
    
    # Capture video
    cap = cv2.VideoCapture(camera_index)
    start_time = time.time()
    duration = 5  # seconds to check

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            return "Error: Cannot read from camera."

 
    
    return "Real Camera Detected (Video stream is active)."

# Replace 0 with your camera index
camera_index = 0
result = check_camera_source(camera_index)
print(result)

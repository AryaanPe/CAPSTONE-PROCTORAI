import numpy as np
import cv2 as cv
import os
import pyautogui
import time
import keyboard
from contextlib import contextmanager

pyautogui.FAILSAFE = False

@contextmanager
def camera_capture(camera_index=0):
    cap = cv.VideoCapture(camera_index)
    try:
        yield cap
    finally:
        cap.release()

def process_eye_region(frame, eyes, eye_size=(100, 50)):
    if len(eyes) > 0:
        (ex, ey, ew, eh) = eyes[0]

        # Remove eyebrow region
        ey += int(0.2 * eh)  # Skip the top 30% for eyebrow
        eh = int(0.8 * eh)  # Keep 70% of the height

        if eh <= 0 or ey + eh > frame.shape[0]:
            return None, frame

        cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_region = frame[ey:ey + eh, ex:ex + ew]

        if eye_region.size == 0:
            return None, frame

        eye_region_gray = cv.cvtColor(eye_region, cv.COLOR_BGR2GRAY)
        eye_region_resized = cv.resize(eye_region_gray, eye_size)

        return eye_region_resized, frame

    return None, frame

def captureEyeAndMouse(target_images=500, max_time=200, folder="eyes", user_sessions=1, 
                       eye_cascade_path=None, eye_size=(100, 50)):
    os.makedirs(folder, exist_ok=True)

    if eye_cascade_path is None:
        eye_cascade_path = cv.data.haarcascades + 'haarcascade_eye.xml'

    eye_cascade = cv.CascadeClassifier(eye_cascade_path)

    if eye_cascade.empty():
        print("Error loading Haar Cascade file.")
        return

    images = []
    mouse_coords = []
    screen_width, screen_height = pyautogui.size()
    step = 125
    current_x, current_y = 0, 0
    direction = 1
    center_threshold = 0.2
    start_time = time.time()
    k_pressed_count = 0

    with camera_capture() as webcam:
        while len(images) < target_images and k_pressed_count < user_sessions:
            if time.time() - start_time > max_time:
                print("Max time reached, stopping capture.")
                break

            if keyboard.is_pressed('k'):
                k_pressed_count += 1
                print(f"K key pressed {k_pressed_count}/{user_sessions}. Pausing capture.")
                keyboard.wait('l')
                print("L key pressed, resuming capture.")

            if k_pressed_count >= user_sessions:
                print("Reached the target number of users, stopping capture.")
                break

            pyautogui.moveTo(current_x, current_y)
            time.sleep(0.05)

            ret, frame = webcam.read()
            if not ret:
                continue

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            processed_eye, frame = process_eye_region(frame, eyes, eye_size)

            if processed_eye is not None:
                is_near_edge = (
                    current_x < screen_width * center_threshold or 
                    current_x > screen_width * (1 - center_threshold) or
                    current_y < screen_height * center_threshold or 
                    current_y > screen_height * (1 - center_threshold)
                )
                num_pics = 2 if is_near_edge else 2

                for _ in range(num_pics):
                    images.append(processed_eye)
                    mouse_coords.append(pyautogui.position())

            cv.imshow("Eye Capture", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            current_x += step * direction
            if current_x >= screen_width or current_x <= 0:
                direction *= -1
                current_y += step

                if current_y >= screen_height:
                    # Teleport to top-left, wait for 1 second, and increase step
                    current_x, current_y = 0, 0
                    step += 25
                    time.sleep(1)
                elif current_y <= 0:
                    current_y = 0
                    step += 25

    cv.destroyAllWindows()

    images_np = np.array(images)
    mouse_coords_np = np.array(mouse_coords)

    try:
        np.save(os.path.join(folder, "eye_images.npy"), images_np)
        np.save(os.path.join(folder, "mouse_coords.npy"), mouse_coords_np)
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    captureEyeAndMouse(target_images=2500, max_time=600, folder="eyes", user_sessions=3)

import cv2
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import keyboard
from sklearn.preprocessing import MinMaxScaler
pyautogui.FAILSAFE = False
# Define the EyeTrackingModel here
class EyeTrackingModel(nn.Module):
    def __init__(self):
        super(EyeTrackingModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(16 * 25 * 12, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Output layer for x and y coordinates

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def load_model(model_path):
    model = EyeTrackingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_eye_image(eye_image):
    # Ensure the image is grayscale
    if len(eye_image.shape) > 2:
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 100x50 as in the training preprocessing
    resized = cv2.resize(eye_image, (50, 100))
    
    # Enhance contrast
    enhanced = enhance_contrast(resized)
    
    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(enhanced.reshape(-1, 1)).reshape(100, 50)
    
    return normalized

def predict_gaze(model, eye_image):
    preprocessed = preprocess_eye_image(eye_image)
    input_tensor = torch.FloatTensor(preprocessed).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().numpy()

def map_gaze_to_screen(gaze_coords, screen_size):
    x, y = gaze_coords
    screen_w, screen_h = screen_size
    screen_x = int(x * screen_w)*1.1
    screen_y = int(y * screen_h)*1.1
    return screen_x, screen_y
def real_time_eye_tracking():
    model = load_model('eye_tracking_model55.pth')
    
    # Load Haar Cascade for eye detection
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    if eye_cascade.empty():
        print("Error loading Haar Cascade file.")
        return
    
    webcam = cv2.VideoCapture(0)
    screen_size = pyautogui.size()
    
    # For smoothing mouse movement
    smoothing_factor = 0.3
    prev_x, prev_y = pyautogui.position()
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]  # Assume the first detected eye is the one we want
            
            # Adjust the eye region to cut out eyebrows
            eyebrow_cut = 10  # You can adjust this value based on how much of the eyebrow to cut out
            ey_adjusted = ey + eyebrow_cut  # Move down to exclude the eyebrow
            eh_adjusted = eh - eyebrow_cut  # Reduce height to avoid eyebrow area
            
            # Ensure the adjusted height is not negative
            if eh_adjusted > 0:
                cv2.rectangle(frame, (ex, ey_adjusted), (ex + ew, ey_adjusted + eh_adjusted), (0, 255, 0), 2)
                
                left_eye = gray_frame[ey_adjusted:ey_adjusted + eh_adjusted, ex:ex + ew]
                if left_eye.size > 0:
                    gaze_coords = predict_gaze(model, left_eye)
                    screen_x, screen_y = map_gaze_to_screen(gaze_coords, screen_size)
                    
                    # Apply smoothing
                    smooth_x = int(prev_x + smoothing_factor * (screen_x - prev_x))
                    smooth_y = int(prev_y + smoothing_factor * (screen_y - prev_y))
                    
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y
                else:
                    print("No eye region captured.")
            else:
                print("Adjusted eye height is invalid.")
        else:
            print("No eyes detected.")
        
        # Debugging output
        if 'gaze_coords' in locals():
            print("Gaze Output:", gaze_coords)  # Check what the model is predicting

        cv2.imshow("Eye Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('k'):
            print("Stopping eye tracking.")
            break
    
    webcam.release()
    cv2.destroyAllWindows()

def load_model(model_path):
    model = EyeTrackingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_eye_image(eye_image):
    # Ensure the image is grayscale
    if len(eye_image.shape) > 2:
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 100x50 as in the training preprocessing
    resized = cv2.resize(eye_image, (50, 100))
    
    # Enhance contrast
    enhanced = enhance_contrast(resized)
    
    # Normalize using MinMaxScaler
    image_scaler = MinMaxScaler()
    normalized = image_scaler.fit_transform(enhanced.reshape(-1, 1)).reshape(100, 50)
    
    return normalized

def predict_gaze(model, eye_image):
    preprocessed = preprocess_eye_image(eye_image)
    input_tensor = torch.FloatTensor(preprocessed).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().numpy()



def real_time_eye_tracking():
    model = load_model('eye_tracking_model55.pth')
    
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    if eye_cascade.empty():
        print("Error loading Haar Cascade file.")
        return
    
    webcam = cv2.VideoCapture(0)
    screen_size = pyautogui.size()
    
    # For smoothing mouse movement
    smoothing_factor = 0.3
    prev_x, prev_y = pyautogui.position()
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]  # Assume the first detected eye is the one we want
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            left_eye = gray_frame[ey:ey + eh, ex:ex + ew]
            if left_eye.size > 0:
                gaze_coords = predict_gaze(model, left_eye)
                screen_x, screen_y = map_gaze_to_screen(gaze_coords, screen_size)
                
                # Apply smoothing
                smooth_x = int(prev_x + smoothing_factor * (screen_x - prev_x))
                smooth_y = int(prev_y + smoothing_factor * (screen_y - prev_y))
                
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y
            else:
                print("No eye region captured.")
        else:
            print("No eyes detected.")
        # Inside your predict_gaze function, after getting the output
        print("Gaze Output:", gaze_coords)  # Check what the model is predicting

        cv2.imshow("Eye Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('k'):
            print("Stopping eye tracking.")
            break
    
    webcam.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    real_time_eye_tracking()
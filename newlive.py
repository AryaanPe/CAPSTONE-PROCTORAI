import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg19 import decode_predictions
import face_recognition
from scipy.spatial.distance import cosine
import threading
import time

# Load VGG19 model for feature extraction
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

# Load liveness detection model
live_model = load_model('bestnew.h5')

# Preprocessing functions
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))   
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_vgg19_features(img):
    img = preprocess_image(img)
    features = vgg_model.predict(img)
    return features.flatten()

# Load and process the reference image for face recognition
reference_image = face_recognition.load_image_file("r (1).jpg")
reference_face_locations = face_recognition.face_locations(reference_image)
reference_face_encodings = face_recognition.face_encodings(reference_image, reference_face_locations)

if len(reference_face_encodings) == 0:
    print("No faces found in the reference image!")
    exit()

reference_face_encoding = reference_face_encodings[0]
reference_features = get_vgg19_features(reference_image[reference_face_locations[0][0]:reference_face_locations[0][2], reference_face_locations[0][3]:reference_face_locations[0][1]])

# Initialize variables
counter = 0
face_match = False
match_count = 0
live_predictions = []
lock = threading.Lock()

def check_face(frame):
    global face_match, match_count

    try:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([reference_face_encoding], face_encoding)

            if match[0]:
                face_image = frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
                face_features = get_vgg19_features(face_image)
                similarity = 1 - cosine(reference_features, face_features)
                
                if similarity > 0.7:  # Similarity threshold
                    with lock:
                        face_match = True
                        match_count += 1
                else:
                    with lock:
                        face_match = False
                        match_count = 0
                return
        
        with lock:
            face_match = False
            match_count = 0
    except ValueError:
        with lock:
            face_match = False
            match_count = 0

def check_liveness(frame):
    global live_predictions

    try:
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            face_region = frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
            face_input = preprocess_image(face_region)
            pred = live_model.predict(face_input)
            live_predictions.append(pred[0][0])
            
            if len(live_predictions) > 20:
                live_predictions.pop(0)

            avg_prediction = np.mean(live_predictions)
            return avg_prediction
        return None
    except Exception as e:
        print(f"Error in liveness check: {e}")
        return None

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if counter % 30 == 0:  # Run every second (assuming 30 FPS)
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
            avg_live_score = check_liveness(frame)
            if avg_live_score is not None:
                if avg_live_score > 0.5:
                    live_label = "REAL"
                    live_color = (0, 255, 0)  # Green
                else:
                    live_label = "FAKE"
                    live_color = (0, 0, 255)  # Red
            else:
                live_label = "Unknown"
                live_color = (255, 255, 255)  # White
            
        with lock:
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if match_count >= 4:
                    match_count = 0  
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.putText(frame, f"Liveness: {live_label}", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, live_color, 2)

        counter += 1
        cv2.imshow("Face Liveness Detection and Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

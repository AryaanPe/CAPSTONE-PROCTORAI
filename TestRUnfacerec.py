import threading
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('latestnew.h5')

# Function to preprocess input image for liveness detection
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image to match model's expected sizing
  # Normalize pixel values to between 0 and 1
    return img

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load face detection model (Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load reference image for face recognition
reference_image = cv2.imread("apref.jpg")
gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
reference_faces = face_cascade.detectMultiScale(gray_reference_image, scaleFactor=1.1, minNeighbors=5)
reference_face_encodings = []

# Encode reference faces
for (x, y, w, h) in reference_faces:
    ref_face = gray_reference_image[y:y+h, x:x+w]
    reference_face_encodings.append(ref_face)

# Initialize face recognizer (e.g., LBPH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(reference_face_encodings, np.array([0] * len(reference_face_encodings)))

# Variables for face recognition
counter = 0
face_match = False
match_count = 0
lock = threading.Lock()

# Function to check face recognition using OpenCV face recognizer
def check_face(frame):
    global face_match, match_count

    try:
        # Convert frame to grayscale
        gray_frame =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the current frame
        frame_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in frame_faces:
            face_region = gray_frame[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_region)
            if confidence < 50:  # Adjust confidence threshold as needed
                with lock:
                    face_match = True
                    match_count += 1
                return
        with lock:
            face_match = False
            match_count = 0
    except ValueError:
        with lock:
            face_match = False
            match_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascades
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]  # Extract face region from the frame
            face_input = preprocess_image(face_region)  # Preprocess the face image

            # Predict liveness using the model
            face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension here
            pred = model.predict(face_input)
            if pred[0][0] > 0.5:
                label = f"{round(pred[0][0], 2)} - Live and {round(pred[0][1], 2)} other"
                color = (0, 255, 0)  # Green color for "Live"
            else:                                  
                label = f"{round(pred[0][1], 2)} - unLive and {round(pred[0][0], 2)} other"                               
                color = (0, 0, 255)  # Red color for "Spoof"

            # Display the label and rectangle around the face
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Check face recognition every 15 frames
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        with lock:
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if match_count >= 4:
                    if len(reference_face_encodings) >= 4:
                        reference_face_encodings.pop(1)
                    reference_face_encodings.append(frame.copy())
                    match_count = 0  # Reset match count after adding a new reference image
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        counter += 1
        cv2.imshow("Face Liveness Detection and Recognition", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

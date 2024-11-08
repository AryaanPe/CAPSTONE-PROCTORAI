import threading
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import face_recognition
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from scipy.spatial.distance import cosine

# Load the VGG19 model for face embedding extraction
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
model = load_model('bestnew.h5')

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_face_embedding(face_image):
    face_input = preprocess_image(face_image)
    embedding = vgg_model.predict(face_input)
    return embedding

# Load and process the reference image
reference_image = face_recognition.load_image_file("ap (1).jpg")
reference_face_locations = face_recognition.face_locations(reference_image)

if len(reference_face_locations) == 0:
    print("No faces found in the reference image!")
    exit()

# Extract the face region from the reference image
top, right, bottom, left = reference_face_locations[0]
reference_face = reference_image[top:bottom, left:right]

# Generate the embedding for the reference face
reference_face_embedding = get_face_embedding(reference_face)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

counter = 0
face_match = False
match_count = 0
lock = threading.Lock()

def check_face(frame):
    global face_match, match_count

    try:
        # Detect faces in the current frame
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            face_region = frame[top:bottom, left:right]

            # Generate embedding for the detected face
            face_embedding = get_face_embedding(face_region)

            # Calculate similarity using cosine distance
            similarity = 1 - cosine(reference_face_embedding, face_embedding)

            if similarity > 0.7:  # Adjust this threshold as needed
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
    ret, frame = cap.read()

    if ret:
        gray = frame

        faces = face_recognition.face_locations(gray)

        for (top, right, bottom, left) in faces:
            face_region = frame[top:bottom, left:right]
            face_input = preprocess_image(face_region)

            pred = model.predict(preprocess_image(gray))
            if pred[0][0] > 0.3:
                label = f"{round(pred[0][0],2)} - Live and {round(pred[0][1],2)} other"
                color = (0, 255, 0)
            else:
                label = f"{round(pred[0][1],2)} - unLive and {round(pred[0][0],2)} other"
                color = (0, 0, 255)

            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        if counter % 15 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        with lock:
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if match_count >= 4:
                    match_count = 0
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        counter += 1
        cv2.imshow("Face Liveness Detection and Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

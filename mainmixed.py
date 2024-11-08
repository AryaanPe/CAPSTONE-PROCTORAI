import threading
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import face_recognition
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

# Load the models
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
model = load_model('bestnew.h5')

# Initialize constants
LAPLACIAN_THRESH = 20
GAUSSIAN_THRESH = 15
WEIGHTED_SCORE_COEFF_LAPLACIAN = 0.25
WEIGHTED_SCORE_COEFF_GAUSSIAN = 0.75
SCORE_THRESHOLD = 50

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

reference_image = face_recognition.load_image_file("ap (1).jpg")
reference_face_locations = face_recognition.face_locations(reference_image)
reference_face_encodings = face_recognition.face_encodings(reference_image, reference_face_locations)

if len(reference_face_encodings) == 0:
    print("No faces found in the reference image!")
    exit()

reference_face_encoding = reference_face_encodings[0]

counter = 0
face_match = False
match_count = 0
lock = threading.Lock()

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img

def laplacian_artifacts(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    _, mask = cv2.threshold(np.abs(laplacian), LAPLACIAN_THRESH, 255, cv2.THRESH_BINARY)
    artifact_percentage = (np.sum(mask > 0) / mask.size) * 100
    return artifact_percentage

def gaussian_blur_artifacts(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, gaussian_blur)
    _, mask = cv2.threshold(diff, GAUSSIAN_THRESH, 255, cv2.THRESH_BINARY)
    artifact_percentage = (np.sum(mask > 0) / mask.size) * 100
    return artifact_percentage
def check_face(frame):
    global face_match, match_count

    try:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([reference_face_encoding], face_encoding)

            if match[0]:
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
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img
while True:
    ret, frame = cap.read()

    if ret:
        gray = frame

        # Frame rate detection
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_rate_label = f"Frame Rate: {frame_rate:.2f} FPS"

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            face_input = preprocess_image(face_region) 

            pred = model.predict(preprocess_image(gray))
            if pred[0][0] > 0.3:
                label = f"{round(pred[0][0],2)} - Live and {round(pred[0][1],2)} other"
                color = (0, 255, 0)  # Green
            else:
                label = f"{round(pred[0][1],2)} - unLive and {round(pred[0][0],2)} other"
                color = (0, 0, 255)  # Red

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Artifact detection
        laplacian_percentage = laplacian_artifacts(frame.copy())
        gaussian_percentage = gaussian_blur_artifacts(frame.copy())
        weighted_score = (WEIGHTED_SCORE_COEFF_LAPLACIAN * laplacian_percentage +
                          WEIGHTED_SCORE_COEFF_GAUSSIAN * gaussian_percentage)
        score_label = "REAL" if weighted_score < SCORE_THRESHOLD else "FAKE"
        score_color = (0, 255, 0) if score_label == "REAL" else (0, 0, 255)

        # Display weighted score and score label
        weighted_score_str = f"Weighted Score: {weighted_score:.2f} ({score_label})"
        cv2.putText(frame, weighted_score_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
        
        # Face matching status
        if counter % 15 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        with lock:
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if match_count >= 4:
                    match_count = 0
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Display frame rate
        cv2.putText(frame, frame_rate_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        counter += 1
        cv2.imshow("Face Liveness Detection and Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

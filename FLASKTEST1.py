import torch  
import threading
import cv2
import numpy as np
import pandas as pd
import face_recognition
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import path

app = Flask(__name__)
socketio = SocketIO(app)
import os

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg19 import VGG19

model = load_model('bestnew.h5')

import torch

model_weights_path = r"C:\Users\aryaa\Documents\PicsCOLAB\best.pt"

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



try:
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights_path)
    print("YOLOv5 model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading YOLOv5 model: {e}")

pathlib.PosixPath = temp



# Initialize constants
LAPLACIAN_THRESH = 20
GAUSSIAN_THRESH = 15
WEIGHTED_SCORE_COEFF_LAPLACIAN = 0.25
WEIGHTED_SCORE_COEFF_GAUSSIAN = 0.75
SCORE_THRESHOLD = 3.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

reference_image_path = "ap (10).jpg"
excel_file_path = "excelTEST.xlsx"

reference_image = face_recognition.load_image_file(reference_image_path)
reference_face_locations = face_recognition.face_locations(reference_image)
reference_face_encodings = face_recognition.face_encodings(reference_image, reference_face_locations)
current_reference_image_name = reference_image_path.split('/')[-1]

if len(reference_face_encodings) == 0:
    raise RuntimeError("No faces found in the reference image!")

reference_face_encoding = reference_face_encodings[0]

counter = 0
face_match = False
match_count = 0
questions_df = None
current_question_index = 0
question_attempted = []

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
    global face_match, match_count, reference_face_encoding
    
    try:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([reference_face_encoding], face_encoding)
            if match[0]:
                face_match = True
                match_count += 1
                return
        face_match = False
        match_count = 0
    except ValueError:
        face_match = False
        match_count = 0

def generate_frames():
    global counter, face_match, match_count, current_reference_image_name
    while True:
        ret, frame = cap.read()
        gray = frame
        if not ret:
            continue

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_rate_label = f"Frame Rate: {frame_rate:.2f} FPS"

        # Perform YOLOv5 object detection for cellphone
        results = yolov5_model(frame)
        labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        # Draw bounding boxes for detected cellphones
        for i, coord in enumerate(coords):
            if labels[i] == 0:  #   class 0 corresponds to 'cellphone'
                x1, y1, x2, y2, conf = coord
                if conf > 0.5:  # Confidence threshold
                    cv2.rectangle(frame, (int(x1 * frame.shape[1]), int(y1 * frame.shape[0])),
                                  (int(x2 * frame.shape[1]), int(y2 * frame.shape[0])),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"Phone: {conf:.2f}", (int(x1 * frame.shape[1]), int(y1 * frame.shape[0] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Face detection and liveness prediction
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            pred = model.predict(preprocess_image(gray))

            if pred[0][0] > 0.3:
                label = f"{round(pred[0][0], 2)} - Live and {round(pred[0][1], 2)} other"
                color = (0, 255, 0)
            else:
                label = f"{round(pred[0][1], 2)} - unLive and {round(pred[0][0], 2)} other"
                color = (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Perform artifact analysis
        laplacian_percentage = laplacian_artifacts(frame.copy())
        gaussian_percentage = gaussian_blur_artifacts(frame.copy())
        weighted_score = (WEIGHTED_SCORE_COEFF_LAPLACIAN * laplacian_percentage +
                          WEIGHTED_SCORE_COEFF_GAUSSIAN * gaussian_percentage)
        score_label = "REAL" if weighted_score < 4.0 else "FAKE"
        score_color = (0, 255, 0) if score_label == "REAL" else (0, 0, 255)

        weighted_score_str = f"Weighted Score: {weighted_score:.2f} ({score_label})"
        cv2.putText(frame, weighted_score_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)

        if counter % 15 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            if match_count >= 4:
                match_count = 0
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.putText(frame, frame_rate_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_reference_image_name:
            cv2.putText(frame, f"Reference Image: {current_reference_image_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        counter += 1
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
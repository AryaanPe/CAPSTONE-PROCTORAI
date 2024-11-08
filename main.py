import threading
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import face_recognition
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooser
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class FaceLivenessApp(App):
    def build(self):
        self.model = load_model('bestnew.h5')
        self.reference_face_encoding = None

        self.layout = BoxLayout(orientation='vertical')
        self.img = Image()
        self.layout.add_widget(self.img)

        self.filechooser = FileChooser()
        self.filechooser.bind(on_selection=self.load_reference_image)
        self.layout.add_widget(self.filechooser)

        self.button = Button(text="Start Camera")
        self.button.bind(on_press=self.start_camera)
        self.layout.add_widget(self.button)

        return self.layout

    def preprocess_image(self, img):
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        return img

    def load_reference_image(self, filechooser, selection):
        if selection:
            reference_image = face_recognition.load_image_file(selection[0])
            reference_face_locations = face_recognition.face_locations(reference_image)
            reference_face_encodings = face_recognition.face_encodings(reference_image, reference_face_locations)

            if len(reference_face_encodings) == 0:
                print("No faces found in the reference image!")
            else:
                self.reference_face_encoding = reference_face_encodings[0]
                print("Reference image loaded successfully!")

    def start_camera(self, instance):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        if ret:
            gray = frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_region = frame[y:y + h, x:x + w]
                face_input = self.preprocess_image(face_region)

                pred = self.model.predict(self.preprocess_image(gray))
                if pred[0][0] > 0.3:
                    label = f"{round(pred[0][0], 2)} - Live and {round(pred[0][1], 2)} other"
                    color = (0, 255, 0)
                else:
                    label = f"{round(pred[0][1], 2)} - unLive and {round(pred[0][0], 2)} other"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if self.reference_face_encoding is not None:
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                for face_encoding in face_encodings:
                    match = face_recognition.compare_faces([self.reference_face_encoding], face_encoding)
                    if match[0]:
                        cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    else:
                        cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    FaceLivenessApp().run()

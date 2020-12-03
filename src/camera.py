import cv2
import numpy as np
from tensorflow import keras
from time import time
import tensorflow as tf

facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = keras.models.load_model('./tmp/model_new_100')
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]

            face_resized = cv2.resize(face, (96, 96))
            preds = model.predict(face_resized[np.newaxis, :, :, np.newaxis] / 255) * 96
            key_pts = np.reshape(preds, [-1, 2])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            for [x_kp, y_kp] in key_pts:
                x_kp = x_kp / 96 * w
                y_kp = y_kp / 96 * h
                cv2.circle(frame, (int(round(x+x_kp)), int(round(y+y_kp))), radius=5, color=[0, 0, 255], thickness=-1)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

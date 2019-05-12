# MIT License
#
# Copyright (c) 2017 Octavio
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from statistics import mode
import cv2
import keras
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import _thread
import scipy.misc

from src.face.utils.datasets import get_labels
from src.face.utils.inference import detect_faces
from src.face.utils.inference import draw_text
from src.face.utils.inference import draw_bounding_box
from src.face.utils.inference import apply_offsets
from src.face.utils.inference import load_detection_model
from src.face.utils.preprocessor import preprocess_input


class RealTimeVideo:
    def __init__(self, ui_obj):
        print("-> Initiating Face Emotion Recognition Model ...")
        self.run = True
        self.gui = ui_obj
        self.detection_model_path = '../../trained_models/face/detection_models/haarcascade_frontalface_default.xml'
        self.emotion_model_path = '../../trained_models/face/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        self.emotion_labels = get_labels('fer2013')
        self.frame_window = 10
        self.emotion_offsets = (20, 40)
        self.face_detection = load_detection_model(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        # print(self.emotion_classifier.summary())
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.emotion_window = []
        self.npAppend = np.empty([1, 7])

    def start_live(self):
        self.video_capture = cv2.VideoCapture(0)
        print("-> Capturing video at - ", self.video_capture.get(cv2.CAP_PROP_FPS), " fps")
        while self.run:
            bgr_image = self.video_capture.read()[1]
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(self.face_detection, gray_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, self.emotion_target_size)
                    img2 = Image.frombytes('L', (gray_face.shape[1], gray_face.shape[0]),
                                           gray_face.astype('b').tostring())
                    img2 = img2.resize((114, 114))
                    im2 = ImageTk.PhotoImage(image=img2)
                    self.gui.s2.create_image(0, 0, image=im2, anchor='nw')
                except:
                    continue
                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = self.emotion_classifier.predict(gray_face)
                self.npAppend = np.concatenate((emotion_prediction, self.npAppend))
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = self.emotion_labels[emotion_label_arg]
                self.emotion_window.append(emotion_text)
                self.gui.res.set(emotion_text)
                self.update_per(emotion_prediction)
                if len(self.emotion_window) > self.frame_window:
                    self.emotion_window.pop(0)
                try:
                    emotion_mode = mode(self.emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

            img = Image.frombytes('RGB', (rgb_image.shape[1], rgb_image.shape[0]), rgb_image.astype('b').tostring())
            img = img.resize((315, 235))
            im = ImageTk.PhotoImage(image=img)
            self.gui.live_feed.create_image(0, 0, image=im, anchor='nw')

            img1 = Image.frombytes('L', (gray_image.shape[1], gray_image.shape[0]), gray_image.astype('b').tostring())
            img1 = img1.resize((153, 115))
            im1 = ImageTk.PhotoImage(image=img1)
            self.gui.s1.create_image(0, 0, image=im1, anchor='nw')
            if not self.run:
                cv2.VideoCapture.release(self.video_capture)
                self.gui.reset()
                self.gui.status.set("Status : Displaying Final Result")
                self.npAppend = self.npAppend[:-1]
                npAppendTranspose = self.npAppend.T
                mean = npAppendTranspose.mean(axis=1)
                self.update_final_per(mean)
                emotion_label_arg = np.argmax(mean)
                emotion_text = self.emotion_labels[emotion_label_arg]
                self.gui.res.set(emotion_text)
                keras.backend.clear_session()

    def update_per(self, emotion_prediction):
        for i in range(7):
            self.gui.probabilities[i].set(str(int(emotion_prediction[0][i]*100))+"%")

    def update_final_per(self, emotion_prediction):
        for i in range(7):
            self.gui.probabilities[i].set(str(int(emotion_prediction[i]*100))+"%")
            print(i, ") updating ", emotion_prediction[i]*100)

    def stop(self):
        self.run = False



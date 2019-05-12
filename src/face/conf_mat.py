import csv
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils.datasets import get_labels
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

detection_model_path = '../trained_models/face/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/face/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
labels = emotion_labels.values()

frame_window = 10
emotion_offsets = (20, 40)

face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]

width, height = 48, 48
given = []
predicted = []

f = open("../datasets/fer2013/fer2013.csv", "r")
dt = csv.DictReader(f)
data = [row for row in dt if row['Usage'] == 'PrivateTest']
print("Testing ", len(data), " images...")
for d in data:
    given_emotion = d.get('emotion')
    pxl = [int(i) for i in d.get('pixels').split(" ")]
    img = np.asarray(pxl).reshape(width, height)
    img = cv2.resize(img.astype(np.uint8), (width, height))
    gray_face = img
    try:
        gray_face = cv2.resize(gray_face, emotion_target_size)
    except:
        continue
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    given.append(int(given_emotion))
    predicted.append(int(emotion_label_arg))

cm = confusion_matrix(given, predicted)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
confmat = []
for r in cm:
    rsum = sum(r)
    confmat.append([float(r[i] / rsum) for i in range(7)])
for r in confmat:
    for val in r:
        print("{0:.2f}".format(val), end=" ")
    print("\n")
plt.show()



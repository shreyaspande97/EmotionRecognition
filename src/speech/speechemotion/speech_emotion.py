import pandas as pd
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json


class SpeechEmotion:

    def __init__(self, ui_obj):
        self.gui = ui_obj
        print("-> Initiating Speech Emotion Recognition Model ...")
        self.trained_model = "../../trained_models/speech/Emotion_Voice_Detection_Model.h5"
        self.model_arch = "../../trained_models/speech/model.json"
        self.labels = {0 : 'Angry', 1 : 'Neutral', 2 : 'Fear', 3 : 'Happy', 4 : 'Sad'}
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        json_file = open(self.model_arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        print(self.loaded_model.summary())
        self.loaded_model.load_weights(self.trained_model)

    def predict(self, wavfile="../audiorecording/output10.wav"):
        print("-> Processing wav file to extract emotion label ...")
        # wavfile = "../speech/audiorecording/youre-so-funny-1.wav"
        data, sample_rate = librosa.load(wavfile, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13), axis=0)
        livedf2 = pd.DataFrame(data=mfccs)
        livedf2 = livedf2.stack().to_frame().T
        twodim = np.expand_dims(livedf2, axis=2)
        livepreds = self.loaded_model.predict(twodim, batch_size=32, verbose=1)
        print(livepreds)
        upd_res = self.update_final_per(livepreds)
        livepreds1 = livepreds.argmax(axis=1)
        id = livepreds1[0] % 5
        self.gui.speechres.set(self.labels[id])
        keras.backend.clear_session()
        output_in_format = []
        output_in_format.append(upd_res[0])
        output_in_format.append(0.0)
        output_in_format.append(upd_res[2])
        output_in_format.append(upd_res[3])
        output_in_format.append(upd_res[4])
        output_in_format.append(0.0)
        output_in_format.append(upd_res[1])
        return output_in_format

    def plot_wave(self, wavfile = "../audiorecording/output10.wav"):
        data, sampling_rate = librosa.load(wavfile)
        plt.figure(figsize=(15, 5))
        print(len(data), "=", data, ",", sampling_rate)
        librosa.display.waveplot(data, sr=sampling_rate)
        plt.show()

    def update_final_per(self, emotion_prediction):
        updated = []
        for i in range(5):
            if emotion_prediction[0][i] > emotion_prediction[0][5+i]:
                temp = emotion_prediction[0][i]
            else:
                temp = emotion_prediction[0][5+i]
            updated.append(temp*100)
            print(i, ") updating ", temp*100)
        fear_val = updated[2]
        for i in range(5):
            if i != 2:
                temp = fear_val * 0.1
                updated[i] = updated[i] + temp
                fear_val = fear_val - temp
                updated[2] = fear_val
        for i in range(5):
            self.gui.speechProb[i].set(str(int(updated[i]))+"%")
        return updated

		
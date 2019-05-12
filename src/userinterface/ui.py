from tkinter import *
from tkinter import font
from tkinter.ttk import Separator
from threading import *
from src.face.utils.datasets import get_labels

from src.face.video_emotion import RealTimeVideo
from src.speech.speechemotion.speech_emotion import SpeechEmotion
from src.speech.audiorecording.rec import Recorder

class GUI(Frame):
    def __init__(self, master):
        super(GUI, self).__init__(master)
        self.recorder = Recorder()
        self.start_live_text = StringVar()
        self.status = StringVar()
        self.speech_status = StringVar()
        self.final_but_text = StringVar()
        self.final_res_text = StringVar()
        self.res = StringVar()
        self.speechres = StringVar()
        self.probabilities = []
        self.speechProb = []
        self.wavfile = 'temp_audio.wav'
        for i in range(7):
            self.probabilities.append(StringVar())
            self.speechProb.append(StringVar())
        self.speech_labels = ['Angry', 'Neutral', 'Fear', 'Happy', 'Sad']
        self.create_ui()
        self.reset()
        self.grid()
        master.protocol('WM_DELETE_WINDOW', self.exit_procedure)

    def exit_procedure(self):
        self.recorder.delete()
        self.quit()

    def startRecording(self):
        self.realtime_thread = Thread(name="Face", target=self.run_face_realtime)
        if self.status.get() == "Status : Running in real time":
            self.video_final_output = self.emo_rec.stop()
            self.wavfile = self.recorder.stop()
        else:
            self.realtime_thread.start()
            self.recorder.start()

    def run_face_realtime(self):
        self.start_live_text.set("Stop")
        self.status.set("Status : Running in real time")
        self.emo_rec = RealTimeVideo(self)
        self.emo_rec.start_live()

    def initiate_speech_post_processing(self):
        self.speech_thread = Thread(name="Speech", target=self.run_speech_post_processing())
        self.speech_thread.start()


    def run_speech_post_processing(self):
        self.speechEmotionRec = SpeechEmotion(self)
        self.speech_probabilities = self.speechEmotionRec.predict(self.wavfile)
        print()

    def integrate(self):
        print("\nIntegration")
        face_acq = 100
        speech_acq = 50
        face_final_output = []
        speech_final_output = []
        final_output = []
        for i in range(7):
            face_final_output.append(int(int(self.probabilities[i].get().strip('%')) * (face_acq/100)))
            speech_final_output.append(int(int(self.speech_probabilities[i]) * (speech_acq/100)))
            final_output.append((face_final_output[i]+speech_final_output[i])/2)
            print(i, ") ", face_final_output[i], " + ", speech_final_output[i], " = ", final_output[i])
        max = 0
        maxpos = 0
        for i in range(7):
            if final_output[i] > max:
                max = final_output[i]
                maxpos = i
        self.final_res_text.set(labels[maxpos])

    def reset(self):
        for i in range(7):
            self.probabilities[i].set("00%")
            self.speechProb[i].set("00%")
        self.res.set("None")
        self.speechres.set("None")
        self.final_but_text.set("Show Final Result")
        self.final_res_text.set("None")
        self.speech_status.set("Status : None")
        self.status.set("Status :")
        self.start_live_text.set("Start Recording")
        self.live_feed.delete("all")
        self.s1.delete("all")
        self.s2.delete("all")

    def create_ui(self):
        header = Label(self, text="Emotion Recognition using Facial Expressions and Speech", font=h1)
        header.grid(row=0, column=0, pady=10)
        Separator(self, orient=HORIZONTAL).grid(row=1, column=0, sticky=EW)
        Separator(self, orient=VERTICAL).grid(row=0, column=1, rowspan=3, sticky=NS)
        master_frame = Frame(self)
        master_frame.grid(row=2, column=0)
        face_frame = Frame(master_frame)
        face_frame.grid(row=0, column=0, rowspan=2, sticky=NW)
        Separator(master_frame, orient=VERTICAL).grid(row=0, column=1, rowspan=2, sticky=NS)
        speech_frame = Frame(master_frame)
        speech_frame.grid(row=0, column=2, sticky=NW)
        Label(face_frame, text="Facial Expression", font=h2).grid(row=0, column=0)
        Label(speech_frame, text="Speech", font=h2).grid(row=0, column=0)
        Separator(face_frame, orient=HORIZONTAL).grid(row=1, column=0, sticky=EW)
        Separator(speech_frame, orient=HORIZONTAL).grid(row=1, column=0, sticky=EW)

        frame_1 = Frame(face_frame)
        frame_1.grid(row=2, column=0, sticky=NW)
        self.live_feed = Canvas(frame_1, width=310, height=230, bg="Black")
        self.live_feed.grid(row=0, column=0, padx=50)

        Separator(frame_1, orient=VERTICAL).grid(row=0, column=1, sticky=NS)

        frame_12 = Frame(frame_1)
        frame_12.grid(row=0, column=2, sticky=N)
        Label(frame_12, text="Result", font=h2).grid(row=0, column=0, columnspan=2, sticky=NSEW)
        Separator(frame_12, orient=HORIZONTAL).grid(row=1, column=0, columnspan=2, sticky=EW)

        frame_121 = Frame(frame_12)
        frame_121.grid(row=2, column=0, sticky=N)
        Label(frame_121, text=labels[0], font=fstd).grid(row=2, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=3, column=0, sticky=EW)
        Label(frame_121, text=labels[1], font=fstd).grid(row=4, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=5, column=0, sticky=EW)
        Label(frame_121, text=labels[2], font=fstd).grid(row=6, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=7, column=0, sticky=EW)
        Label(frame_121, text=labels[3], font=fstd).grid(row=8, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=9, column=0, sticky=EW)
        Label(frame_121, text=labels[4], font=fstd).grid(row=10, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=11, column=0, sticky=EW)
        Label(frame_121, text=labels[5], font=fstd).grid(row=12, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=13, column=0, sticky=EW)
        Label(frame_121, text=labels[6], font=fstd).grid(row=14, column=0, sticky='w')


        frame_122 = Frame(frame_12)
        frame_122.grid(row=2, column=1, sticky=N, padx=5)
        Label(frame_122, textvariable=self.probabilities[0], font=fstd).grid(row=2, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=3, column=0, sticky=EW)
        Label(frame_122, textvariable=self.probabilities[1], font=fstd).grid(row=4, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=5, column=0, sticky=EW)
        Label(frame_122, textvariable=self.probabilities[2], font=fstd).grid(row=6, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=7, column=0, sticky=EW)
        Label(frame_122, textvariable=self.probabilities[3], font=fstd).grid(row=8, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=9, column=0, sticky=EW)
        Label(frame_122, textvariable=self.probabilities[4], font=fstd).grid(row=10, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=11, column=0, sticky=EW)
        Label(frame_122, textvariable=self.probabilities[5], font=fstd).grid(row=12, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=13, column=0, sticky=EW)
        Label(frame_122, textvariable=self.probabilities[6], font=fstd).grid(row=14, column=0, sticky='w')

        Separator(frame_12, orient=HORIZONTAL).grid(row=3, column=0, columnspan=2, sticky=EW)
        Label(frame_12, textvariable=self.res, font=h2).grid(row=4, column=0, columnspan=2)

        Separator(face_frame, orient=HORIZONTAL).grid(row=3, column=0, sticky=EW)

        Label(face_frame, text="Stages", font=h3).grid(row=4, column=0, sticky='new')
        Label(face_frame, text="", font=h3).grid(row=5, column=0, sticky='new')

        frame_2 = Frame(face_frame)
        frame_2.grid(row=6, column=0, sticky=NSEW)
        self.s1 = Canvas(frame_2, width=150, height=112, bg="Black")
        self.s1.grid(row=0, column=0, padx=70)
        self.s2 = Canvas(frame_2, width=112, height=112, bg="Black")
        self.s2.grid(row=0, column=1, padx=70)
        Label(frame_2, text="Preprocessing", font=fstd).grid(row=1, column=0)
        Label(frame_2, text="Face Detection", font=fstd).grid(row=1, column=1)

        Separator(face_frame, orient=HORIZONTAL).grid(row=7, column=0, sticky=EW)

        Label(face_frame, textvariable=self.status, font=fstd).grid(row=8, column=0, padx=20, pady=5, sticky=NW)

        Separator(face_frame, orient=HORIZONTAL).grid(row=9, column=0, sticky=EW)

        frame_3 = Frame(face_frame)
        frame_3.grid(row=10, column=0, sticky=EW, padx=20, pady=10)
        self.start_live = Button(frame_3, textvariable=self.start_live_text, command=self.startRecording)
        self.start_live.grid(row=0, column=0, padx=200)


        frame_1 = Frame(speech_frame)
        frame_1.grid(row=2, column=0, sticky=NW)
        self.status_frame = Frame(frame_1, width=50, height=15)
        self.status_frame.grid(row=0, column=0, padx=20)

        self.start_live = Button(self.status_frame, text="Predict emotion from wav file", command=self.initiate_speech_post_processing)
        self.start_live.grid(row=0, column=0, pady=20, sticky=NSEW)

        # Label(self.status_frame, textvariable=self.speech_status, font=fstd).grid(row=1, column=0, sticky=NSEW)

        Separator(frame_1, orient=VERTICAL).grid(row=0, column=1, sticky=NS)

        frame_12 = Frame(frame_1)
        frame_12.grid(row=0, column=2, sticky=N)
        Label(frame_12, text="Result", font=h2).grid(row=0, column=0, columnspan=2, sticky=NSEW)
        Separator(frame_12, orient=HORIZONTAL).grid(row=1, column=0, columnspan=2, sticky=EW)

        frame_121 = Frame(frame_12)
        frame_121.grid(row=2, column=0, sticky=N)
        Label(frame_121, text=self.speech_labels[0], font=fstd).grid(row=2, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=3, column=0, sticky=EW)
        Label(frame_121, text=self.speech_labels[1], font=fstd).grid(row=4, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=5, column=0, sticky=EW)
        Label(frame_121, text=self.speech_labels[2], font=fstd).grid(row=6, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=7, column=0, sticky=EW)
        Label(frame_121, text=self.speech_labels[3], font=fstd).grid(row=8, column=0, sticky='w')
        Separator(frame_121, orient=HORIZONTAL).grid(row=9, column=0, sticky=EW)
        Label(frame_121, text=self.speech_labels[4], font=fstd).grid(row=10, column=0, sticky='w')

        frame_122 = Frame(frame_12)
        frame_122.grid(row=2, column=1, sticky=N, padx=5)
        Label(frame_122, textvariable=self.speechProb[0], font=fstd).grid(row=2, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=3, column=0, sticky=EW)
        Label(frame_122, textvariable=self.speechProb[1], font=fstd).grid(row=4, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=5, column=0, sticky=EW)
        Label(frame_122, textvariable=self.speechProb[2], font=fstd).grid(row=6, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=7, column=0, sticky=EW)
        Label(frame_122, textvariable=self.speechProb[3], font=fstd).grid(row=8, column=0, sticky='w')
        Separator(frame_122, orient=HORIZONTAL).grid(row=9, column=0, sticky=EW)
        Label(frame_122, textvariable=self.speechProb[4], font=fstd).grid(row=10, column=0, sticky='w')

        Separator(frame_12, orient=HORIZONTAL).grid(row=3, column=0, columnspan=2, sticky=EW)
        Label(frame_12, textvariable=self.speechres, font=h2).grid(row=4, column=0, columnspan=2)

        Separator(speech_frame, orient=HORIZONTAL).grid(row=3, column=0, sticky=EW)

        Separator(self, orient=HORIZONTAL).grid(row=4, column=0, sticky=EW)

        final_frame = Frame(speech_frame)
        final_frame.grid(row=5, column=0, sticky=EW)

        Label(final_frame, text="Final Result", font=h2).grid(row=0, column=0)
        Separator(final_frame, orient=HORIZONTAL).grid(row=1, column=0, sticky=EW)
        self.final_results = Button(final_frame, textvariable=self.final_but_text, command=self.integrate)
        self.final_results.grid(row=2, column=0, pady=30)
        Label(final_frame, textvariable=self.final_res_text, font=h2).grid(row=3, column=0, padx=139)


if __name__ == "__main__":

    labels = get_labels('fer2013')
    res = [0, 0, 0, 0, 0, 0, 0]
    root = Tk()
    root.title("Emotion Recognition")
    h1 = font.Font(family="Courier", size=15, weight="bold")
    h2 = font.Font(family="Courier", size=13, weight="bold")
    h3 = font.Font(family="Courier", size=13)
    fstd = font.Font(family="Courier", size=12)
    root.bind('<Escape>', lambda e: root.quit())
    app = GUI(root)
    root.mainloop()

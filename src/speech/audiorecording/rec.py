import pyaudio
import wave
import subprocess
import os
import time
import threading


class Recorder:

    def __init__(self, chunk=1024, channels=2, rate=44100):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

    def start(self, wavfile="temp_audio.wav"):
        self.wavfile = wavfile
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print("-> Started recording audio")
        while self._running:
            data = stream.read(self.CHUNK)
            self._frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("-> Audio recording completed")
        self.save(self.wavfile)

    def stop(self):
        self._running = False
        return self.wavfile

    def save(self, filename):
        print("-> Saving audio to " + filename)
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("-> File saved")

    @staticmethod
    def delete(filename=""):
        if os.path.isfile(filename):
            os.remove(filename)

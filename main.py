#!/usr/bin/env python3.5

import hatesonar as hs
import matplotlib.pyplot as plt
import multiprocessing
import os
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
import speech_recognition as sr
import scipy.io.wavfile as wavfile
import subprocess
import sys


mic = sr.Microphone()
mic.CHUNK = 4096
r = sr.Recognizer()


def splitAudio(inputFile):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [fs, x] = audioBasicIO.readAudioFile(inputFile)
    segmentLimits = aS.silenceRemoval(x, fs, 0.05, 0.05,
                                      1.0, 0.3, False)
    for i, s in enumerate(segmentLimits):
        strOut = "./data/audio_{0:.3f}-{1:.3f}.wav".format(s[0], s[1])
        wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])

def get_audio_from_file(inputFile):
    audioFile = sr.AudioFile(inputFile)
    with audioFile as source:
        audio = r.record(source)
    return audio

def hatepercent(text):
    sonar = hs.Sonar()
    x = sonar.ping(text=text)
    return x['classes'][0]['confidence']


def hplot(x_list, y_list):
    try:
        if y_list[-1] < 0.2:
            col = 'green'
        elif y_list[-1] < 0.4:
            col = 'orange'
        else: col = 'red'
    except:
        col = 'green'
    plt.plot(x_list, y_list, color = col)
    plt.ylabel("hate %")
    plt.xlabel("phrase number")
    plt.draw()
    plt.pause(0.05)


def rec():
    with mic as source:
        audio = r.listen(source)
    return audio


def process_speech(buff):
    i = 1
    x_axis = []
    y_axis = []
    while True:
        audio = buff.get()
        try:
            x = r.recognize_google(audio)
            print(x)
        except:
            print("Sorry, didn't caught that, try again")
            x = ""
        hate = hatepercent(x)
        x_axis.append(i)
        y_axis.append(hate)
        hplot(x_axis, y_axis)
        i += 1
    plt.show()

   
def rec_from_file():
    try:
        fileName = sys.argv[2]
    except:
        print("usage: ./main file <filename>") 

    splitAudio(fileName)
    audio_file_list = os.listdir('./data/')
    audio_file_list.sort()
    for f in audio_file_list:
        ff = './data/' + f
        buff.put(get_audio_from_file(ff))
        
    for f in audio_file_list:
        ff = './data/' + f
        os.remove(ff)

def rec_from_video():
    try:
        fileName = sys.argv[2]
        print("sys.argv[2]: " + fileName)
    except:
        print("usage: ./main video <filename>")

    command = "ffmpeg -i " + fileName + " -ab 160k -ac 2 -ar 44100 -vn ./audio.wav"
    subprocess.call(command, shell=True) 
    sys.argv[2] = "./audio.wav"
    rec_from_file()
    os.remove('./audio.wav')
    

def rec_from_mic():
    while True:
        buff.put(rec())


def choose_task_and_execute(command):
    tasks = {
                'mic' : rec_from_mic,
                'file' : rec_from_file,
                'video' : rec_from_video
            }

    tasks[command]()


if __name__ == "__main__":
    buff = multiprocessing.Queue()
    proc = multiprocessing.Process(target=process_speech, args=(buff,), )
    proc.start()
    choose_task_and_execute(sys.argv[1])
    print("ALL DONE!")

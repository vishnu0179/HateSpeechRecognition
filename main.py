#!/usr/bin/env python3.5

import hatesonar as hs
import matplotlib.pyplot as plt
import multiprocessing
import speech_recognition as sr
import sys


mic = sr.Microphone()
mic.CHUNK = 4096
r = sr.Recognizer()


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
    

if __name__ == "__main__":
    buff = multiprocessing.Queue()
    proc = multiprocessing.Process(target=process_speech, args=(buff,))
    proc.start()
    while True:
        buff.put(rec())

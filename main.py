#!/usr/bin/env python3.5

import multiprocessing
import speech_recognition as sr
import sys

mic = sr.Microphone()
mic.CHUNK = 4096
r = sr.Recognizer()


def rec():
    with mic as source:
        audio = r.listen(source)
    return audio

def under(buff):
    while True:
        audio = buff.get()
        try:
            x = r.recognize_google(audio)
            print(x)
        except:
            print("Sorry, didn't caught that, try again")
    


if __name__ == "__main__":
    buff = multiprocessing.Queue()
    proc = multiprocessing.Process(target=under, args=(buff,))
    proc.start()
    while True:
        buff.put(rec())

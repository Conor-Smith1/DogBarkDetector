#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import pyaudio
from scipy.fft import rfft
import wave
from datetime import datetime
import http
import urllib
from sklearn.ensemble import RandomForestClassifier
import pickle
import http.client

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning)


numBins = 256
Window= 131072
Step= 10000
labelVal= 16000


#A fast function for computing FFTs split up into bins
def fastMeanBins(data,bins):
    realfft=np.abs(rfft(data))
    realfft[0] = 0
    length =len(realfft)
    binWidth = int(length/bins)
    if length%binWidth == 0:
        RSarray = realfft.reshape(-1,binWidth)
        binnedFFT = np.mean(RSarray,axis = 1)
    else:
        realfft = realfft[:(length//binWidth)*binWidth]
        RSarray = realfft.reshape(-1,binWidth)
        binnedFFT = np.mean(RSarray,axis = 1)
    binnedFFT = list(binnedFFT)
    return binnedFFT


# the file name output you want to record into
filename = "test.wav"
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 44100
record_seconds = 3.1
# initialize PyAudio object
p = pyaudio.PyAudio()
# open stream object as input & output
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)


RandomForestModel = pickle.load( open( "RFC.p", "rb" ) )

print("Recording...")
allResults = []

try:
    while True:
        frames = []
        for i in range(int(sample_rate / chunk * record_seconds)):
            data = stream.read(chunk)
            # if you want to hear your voice while recording
            # stream.write(data)
            frames.append(data)
        allFrames = b"".join(frames)
        sig = np.frombuffer(allFrames, dtype='<i2')
        Features = fastMeanBins(sig[:Window],numBins)

        
        result = int(RandomForestModel.predict([Features]))

        if result == 1:
            print('Bark Detected')
            conn = http.client.HTTPSConnection("api.pushover.net:443")
            conn.request("POST", "/1/messages.json",
              urllib.parse.urlencode({
                "token": "aofckodfijgub9hadoqagth4imxscs",
                "user": "u9opohtf96soi4o27w7fip68wn4kjg",
                "message": "Dog Barking",
              }), { "Content-type": "application/x-www-form-urlencoded" })
            conn.getresponse()
            now = datetime.now() # current date and time
            fname = now.strftime("%m-%d-%Y, %H-%M-%S")
            wf = wave.open(fname+'.wav', "wb")
            # set the channels
            wf.setnchannels(channels)
            # set the sample format
            wf.setsampwidth(p.get_sample_size(FORMAT))
            # set the sample rate
            wf.setframerate(sample_rate)
            # write the frames as bytes
            wf.writeframes(b"".join(frames))
            # close the file
            wf.close()
except KeyboardInterrupt:
    pass  
    
    
print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()
# save audio file
# open the file in 'write bytes' mode




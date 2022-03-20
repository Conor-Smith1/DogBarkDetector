#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from scipy.io import wavfile
import csv
import numpy as np
from scipy.fft import rfft
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

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

def buildLabels(soundData,labelData):
    totalLength = len(soundData)
    labels = np.zeros(totalLength)
    for labelEx in labelData:
        start = round(float(labelEx[0])*44100)
        end = round(float(labelEx[1])*44100)
        labels[start:end] = 1
    return labels

def createFeatureData(soundData,labels,window = 44100,step = 10000, samplerate = 44100,labelVal = 16000,numBins = 256):
    
    columnNames = ['FFTBin'+str(x) for x in range(numBins)]
    columnNames.insert(0,'label')
    combData = []
    ls = buildLabels(soundData,labels)
    for d in range(int((len(soundData)-window)/step)):
        start = d*step
        end = d*step+window
        features = fastMeanBins(soundData[start:end],numBins)
        if np.sum(ls[start:end])>labelVal:
            l = 1
        else:
            l = 0
        readyData = np.insert(features,0,l)
        combData.append(readyData)
    dFrame = pd.DataFrame(combData,columns = columnNames)
    return dFrame

fileRoots = []
for f in os.listdir():
    if '.txt' in f:
        fileRoots.append(f.split('.')[0])

featureDict = {}
for file in fileRoots:
    i = []
    rate,d = wavfile.read(file+'.wav')
    with open(file+'.txt', newline = '') as labelData:                                                                                          
            labelReader = csv.reader(labelData, delimiter='\t')
            for label in labelReader:
                i.append(label)
    print(file)
    featureDict[file] = createFeatureData(d,i,window = 131072,numBins = 256)

finalDF = pd.concat(featureDict)

y, x = finalDF.iloc[:, [0]], finalDF.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

RFC = RandomForestClassifier()

RFC.fit(x_train, y_train.values.ravel())

print('Full Model score on Training Data:',RFC.score(x_train, y_train))
print('Full Model score on Eval Data:',RFC.score(x_test, y_test))    
#evaluate model and print report

#The below should only be used for actual classification reports NOT for linear regression
y_pred = RFC.predict(x_test)  
report = classification_report(y_test,y_pred)
print(report)

pickle.dump(RFC, open( "RFC.p", "wb" ) )
#test = pickle.load( open( "RFC.p", "rb" ) )


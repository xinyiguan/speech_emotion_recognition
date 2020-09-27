import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#====================Function for feature extraction============================
# extract the mfcc, chroma, and mel features from a sound file.

def function extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype = 'float32')
        sample_rate = sound_file.samplerate

        # If chroma is True, get the Short-Time Fourier Transform of X.
        if chroma:
            stft = np.abs(librosa.stft(X))

        # Create an empty numpy array,
        # for each feature of the three,make a call to the corresponding function from librosa.feature
        # stack the arrays in sequence horizontally, update result
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T), axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T), axis=0)
            result = np.hstack((result, mel))
    return result

#================================ Emotion Dictionary ===========================
# Define a dictionary:
emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprise',
}

#Emotions to observe:
observed_emotions = {'calm', 'happy', 'fearful', 'disgust'}

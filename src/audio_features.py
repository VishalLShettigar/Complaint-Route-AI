import numpy as np
import librosa
import os

def extract_audio_features(path):
    if not path or not os.path.exists(path):
        return np.zeros(13)
    try:
        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.mean(axis=1)
    except:
        return np.zeros(13)
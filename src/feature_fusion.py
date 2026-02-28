import numpy as np

def fuse_features(text, audio, video):
    return np.concatenate([text, audio, video])
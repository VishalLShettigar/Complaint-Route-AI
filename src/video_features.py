import numpy as np
import cv2
import os

def extract_video_features(path):
    if not path or not os.path.exists(path):
        return np.zeros(1)
    cap = cv2.VideoCapture(path)
    values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values.append(gray.mean())
    cap.release()
    return np.array([sum(values)/len(values)]) if values else np.zeros(1)
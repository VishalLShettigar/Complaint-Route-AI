import pandas as pd
import joblib
import faiss
import numpy as np

from src.text_features import extract_text_features
from src.audio_features import extract_audio_features
from src.video_features import extract_video_features
from src.feature_fusion import fuse_features

# Load models
priority_model = joblib.load("models/priority_model.pkl")
eta_model = joblib.load("models/eta_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load data
complaints = pd.read_csv("data/raw/complaints.csv")
officers = pd.read_csv("data/raw/officers.csv")

# Similarity index
index = faiss.read_index("models/similarity.index")

# Officer embeddings
officer_embeddings = np.array(
    [extract_text_features(skill) for skill in officers["skills"]]
)

def route_officer(text_embedding):
    scores = officer_embeddings @ text_embedding
    best_idx = scores.argmax()
    officer = officers.iloc[best_idx]
    return {
        "name": officer["name"],
        "department": officer["department"]
    }

def predict(text, audio_path=None, video_path=None):
    # Feature extraction
    text_f = extract_text_features(text)
    audio_f = extract_audio_features(audio_path)
    video_f = extract_video_features(video_path)

    # Fuse & scale
    fused = fuse_features(text_f, audio_f, video_f)
    fused_scaled = scaler.transform([fused])

    # Predictions
    priority = priority_model.predict(fused_scaled)[0]
    eta_days = int(round(eta_model.predict(fused_scaled)[0]))

    # Officer routing
    officer = route_officer(text_f)

    # Past similar complaints
    _, indices = index.search(fused_scaled, 5)
    past_complaints = complaints.iloc[indices[0]][
        ["text", "priority", "eta_days"]
    ].to_dict(orient="records")

    return {
        "priority": priority,
        "eta_days": max(1, eta_days),  # safety
        "officer": officer,
        "past_complaints": past_complaints
    }
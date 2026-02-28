import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score

from src.text_features import extract_text_features
from src.audio_features import extract_audio_features
from src.video_features import extract_video_features
from src.feature_fusion import fuse_features


def evaluate_models():
    data = pd.read_csv("data/raw/complaints.csv")

    priority_model = joblib.load("models/priority_model.pkl")
    eta_model = joblib.load("models/eta_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X, y_priority, y_eta = [], data["priority"], data["eta_days"]

    for _, row in data.iterrows():
        t = extract_text_features(row["text"])
        a = extract_audio_features(row["audio_path"])
        v = extract_video_features(row["video_path"])
        X.append(fuse_features(t, a, v))

    X = scaler.transform(np.array(X))

    pred_priority = priority_model.predict(X)
    pred_eta = eta_model.predict(X)

    report = classification_report(
        y_priority, pred_priority, output_dict=True
    )

    accuracy = accuracy_score(y_priority, pred_priority)
    mae = mean_absolute_error(y_eta, pred_eta)

    priority_distribution = data["priority"].value_counts().to_dict()

    return {
        "total_complaints": len(data),
        "priority_distribution": priority_distribution,
        "classification_report": report,
        "accuracy": round(accuracy, 3),
        "mae": round(mae, 2)
    }
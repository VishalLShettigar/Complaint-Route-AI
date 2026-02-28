import pandas as pd
import numpy as np
import joblib
import faiss

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.text_features import extract_text_features
from src.audio_features import extract_audio_features
from src.video_features import extract_video_features
from src.feature_fusion import fuse_features

data = pd.read_csv("data/raw/complaints.csv")

X, y_p, y_e = [], data["priority"], data["eta_days"]

for _, row in data.iterrows():
    t = extract_text_features(row["text"])
    a = extract_audio_features(row["audio_path"])
    v = extract_video_features(row["video_path"])
    X.append(fuse_features(t, a, v))

X = np.array(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

priority_model = RandomForestClassifier(n_estimators=200)
eta_model = RandomForestRegressor(n_estimators=200)

priority_model.fit(X, y_p)
eta_model.fit(X, y_e)

joblib.dump(priority_model, "models/priority_model.pkl")
joblib.dump(eta_model, "models/eta_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

index = faiss.IndexFlatL2(X.shape[1])
index.add(X)
faiss.write_index(index, "models/similarity.index")

print("Training completed")
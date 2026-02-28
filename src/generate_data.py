import pandas as pd
import random

high = [
    "Server is down after update",
    "Database crashed and data inaccessible",
    "Ransomware detected on system",
    "Office internet completely down"
]

medium = [
    "Laptop is very slow",
    "Firewall blocking internal app",
    "Cloud VM responding slowly"
]

low = [
    "Need software installation",
    "Password reset request",
    "Email setup needed"
]

rows = []
cid = 1

for _ in range(30):
    rows.append([cid, random.choice(high), "data/raw/audio_samples/high.wav",
                 "data/raw/video_samples/high.mp4", "High", 1, random.randint(1,5)])
    cid += 1

for _ in range(30):
    rows.append([cid, random.choice(medium), "data/raw/audio_samples/medium.wav",
                 "data/raw/video_samples/medium.mp4", "Medium", 3, random.randint(1,5)])
    cid += 1

for _ in range(30):
    rows.append([cid, random.choice(low), "data/raw/audio_samples/low.wav",
                 "data/raw/video_samples/low.mp4", "Low", 6, random.randint(1,5)])
    cid += 1

df = pd.DataFrame(rows, columns=[
    "complaint_id","text","audio_path","video_path",
    "priority","eta_days","officer_id"
])

df.to_csv("data/raw/complaints.csv", index=False)
print("Complaints generated:", len(df))
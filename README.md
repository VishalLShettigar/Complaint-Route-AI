# AI-Based Multimodal Complaint Routing System

## 1. Overview
This project implements an **offline AI/ML-based complaint routing system** designed to automatically analyze IT-related complaints and route them to the appropriate officer.

The system supports **multimodal inputs**:
- Text complaints
- Audio complaints
- Video complaints

For each complaint, the system predicts:
- **Priority level** (High / Medium / Low)
- **Estimated resolution time (ETA in days)**
- **Assigned officer and department**
- **Similar past complaints**
- **Model evaluation metrics (Admin view)**

This project was developed as an **interview assignment**, focusing on clean architecture, correctness, and real-world feasibility.

---

## 2. Problem Statement
In many organizations, IT complaints are handled manually, which leads to:
- Delayed response times
- Incorrect prioritization
- Inefficient officer assignment
- No use of historical complaint data

This project aims to **automate complaint analysis and routing** using machine learning to improve efficiency and decision-making.

---

## 3. System Architecture
The system follows a modular ML pipeline:

User Complaint (Text / Audio / Video)
↓
Feature Extraction
(Text Embeddings, Audio MFCC, Video Statistics)
↓
Feature Fusion
↓
Machine Learning Models
(Priority Classification & ETA Regression)
↓
Officer Routing (Semantic Similarity)
↓
Past Complaint Retrieval (FAISS)
↓
Web Interface + Admin Dashboard

---

## 4. Technologies Used

### Programming Language
- Python 3.9+

### Libraries & Frameworks
- Flask – Web application backend
- Scikit-learn – Machine learning models
- Sentence-Transformers – Text embeddings
- Librosa – Audio feature extraction
- OpenCV – Video feature extraction
- FAISS – Similarity search
- Pandas, NumPy – Data processing
- Joblib – Model persistence

⚠️ The entire system runs **fully offline** with **no external APIs**.

---

## 5. Dataset Description

### Complaints Dataset (`complaints.csv`)
Each record contains:
- Complaint text
- Audio file path (optional)
- Video file path (optional)
- Priority label (High / Medium / Low)
- Estimated resolution time (ETA in days)
- Officer ID

Synthetic but realistic data is used for demonstration and training.

### Officers Dataset (`officers.csv`)
Each officer record contains:
- Officer ID
- Name
- Department
- Skill keywords

Officer assignment is done using **semantic similarity** between complaint text and officer skills.

---

## 6. Feature Engineering

### Text Features
- Sentence-BERT embeddings (`all-MiniLM-L6-v2`)
- Implicit support for multilingual text

### Audio Features
- MFCC (Mel Frequency Cepstral Coefficients)
- Safe fallback for missing or empty audio files

### Video Features
- Frame-level grayscale intensity statistics
- Safe fallback for missing or empty video files

### Feature Fusion
All extracted features are concatenated into a single numerical feature vector.

---

## 7. Machine Learning Models

### Priority Prediction
- RandomForest Classifier
- Predicts: High / Medium / Low

### ETA Prediction
- RandomForest Regressor
- Predicts: number of days required to resolve the complaint

### Officer Routing
- Semantic similarity between complaint embeddings and officer skill embeddings

### Similar Complaint Retrieval
- FAISS similarity search over fused feature vectors

---

## 8. Model Evaluation

### Priority Classification Metrics
- Precision
- Recall
- F1-score
- Overall accuracy

### ETA Regression Metrics
- Mean Absolute Error (MAE)

Evaluation metrics are computed using the available dataset and are displayed in the **Admin Dashboard**.
Due to synthetic data usage, evaluation is performed on the same dataset.
In real-world deployment, a train–validation split would be applied.

---

## 9. Web Application Features

### User Interface
- Submit complaint text
- View predicted priority
- View estimated resolution time (ETA)
- View assigned officer and department
- View similar past complaints

### Admin Dashboard
- View all stored complaints
- View dataset statistics
- View priority distribution
- View full model evaluation metrics (precision, recall, F1, accuracy, MAE)

---

## 10. How to Run the Project

### Step 1: Install dependencies
pip install -r requirements.txt

### Step 2: Generate sample data
python -m src.generate_data
### Step 3: Train machine learning models
python -m src.train_models
### Step 4: Run the web application
python app.py
### Step 5: Access the application

Main application:
http://127.0.0.1:5000

Admin dashboard:
http://127.0.0.1:5000/admin

## 11. Assumptions & Constraints

Synthetic data is used for demonstration

Audio and video inputs are optional

Missing media files are handled safely

System is designed for offline execution

## 12. Limitations

Model accuracy depends on data quality and quantity

Synthetic data limits real-world generalization

Not deployed in a production environment

## 13. Future Enhancements

Training with real complaint data

Advanced deep multimodal models

Real-time audio/video recording

Cloud deployment and authentication

Analytics dashboard for complaint trends

## 14. Conclusion

This project demonstrates a complete end-to-end AI-based multimodal complaint routing system using machine learning.
It showcases practical ML engineering skills, clean architecture, and real-world design considerations suitable for evaluation.
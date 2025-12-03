**FAKE NEWS CLASSIFIER(NLP+MACHINE LEARNING)**

A machine-learning powered system that detects whether a news article is REAL or FAKE using Natural Language Processing (NLP) techniques.
Built using Python, TF-IDF, Logistic Regression, Scikit-learn, and deployed as a prediction script.
Project Overview

Fake news is one of the biggest challenges in today’s digital world.
This project builds a text classification model that predicts whether a given news article is real or fake using machine learning.

**THE WORKFLOW INCLUDES :**
Text preprocessing
TF-IDF vectorization
ML model training
Evaluation
Model saving using Pickle
Real-time prediction script

 **PROJECT STRUCTURE**
 FakeNewsClassifier/
│
├── data/
│     ├── Fake.csv
│     ├── True.csv
│
├── models/
│     ├── fake_news_model.pkl
│     ├── tfidf_vectorizer.pkl
│
├── src/
│     ├── train_model.py
│     ├── predict.py
│     ├── preprocess.py (optional, for custom cleaning)
│
├── venv/  (virtual environment)
│
└── README.md

**DATASET**
Dataset used: Fake and Real News Dataset (Kaggle)
Contains:
Fake.csv → news labeled as fake
True.csv → news labeled as real
Dataset fields:
title
text
subject
date
label (added manually)

**MACHINE LEARNING APPROACH**
1)Data Loading :
Both datasets are merged and labeled:
Fake → 0
Real → 1

2️)TextVectorization :
Used TF-IDF Vectorizer to convert news text into numerical features.
Key settings:
TfidfVectorizer(stop_words='english', max_df=0.7)

3️) Model Training :
Trained using Logistic Regression — one of the best baseline models for NLP classification.

4)Evaluation:
Achieved accuracy:
0.9834075723830735


5️)Saving the Model:
Used Pickle to save:
trained model → fake_news_model.pkl
vectorizer → tfidf_vectorizer.pkl

**CODE SUMMARY**

->Train the Model
Located in: src/train_model.py
Steps:
Load dataset
Preprocess text
Split data
Train model
Evaluate
Save model

->Prediction Script
Located in: src/predict.py
Usage:
Run the script
Enter news text
Receive prediction (REAL or FAKE)

**HOW TO RUN THE PROJECT**
1. Create Virtual Environment
python -m venv venv
Activate:
Windows:
venv\Scripts\activate

2. Install Dependencies
pip install numpy pandas scikit-learn nltk flask

3. Train the Model
python src/train_model.py

4. Make Predictions
python src/predict.py

Example input:
The government has announced that aliens were seen on the moon yesterday.
Output:
Prediction: FAKE

**KEY FEATURES**
Real-world NLP dataset
High accuracy classifier
Clean ML pipeline
Modular code
Saved model for real-time predictions
Ready for deployment as a web app

**TECHNOLOGIES USED**
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Pickle
NLP preprocessing

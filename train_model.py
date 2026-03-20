import pandas as pd
import numpy as np
import os
import requests
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# The dataset is ~30MB, open raw GitHub link for Fake and Real News dataset
DATA_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODEL_DIR = "model"

def download_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        response = requests.get(DATA_URL)
        if response.status_code == 200:
            with open(DATA_PATH, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            raise Exception("Failed to download dataset. Status code:", response.status_code)
    else:
        print("Dataset already exists.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 1. Download & Load Dataset
    download_dataset()
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Dataset columns: 'Unnamed: 0', 'title', 'text', 'label'
    print(f"Dataset Shape: {df.shape}")
    
    # Drop empty rows
    df.dropna(subset=['text', 'label'], inplace=True)
    
    # 2. Text Preprocessing
    print("Cleaning text data... (this may take a moment)")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Features and labels (REAL -> 1, FAKE -> 0)
    X = df['clean_text']
    y = df['label'].apply(lambda x: 1 if x == 'REAL' else 0)
    
    print("Splitting into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying TF-IDF Vectorization...")
    # Using max_df to remove overly frequent words and stop_words='english'
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print("Saved TF-IDF Vectorizer.")
    
    # 3. Model Training - Random Forest
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    rf_predictions = rf_model.predict(X_test_tfidf)
    
    print("\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
    print(classification_report(y_test, rf_predictions, target_names=['FAKE (0)', 'REAL (1)']))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    print("Saved Random Forest Model.")
    
    # 3. Model Training - SVM
    print("\nTraining Support Vector Machine (SVC)...")
    # Linear kernel tends to perform very well for text classification and is faster to train
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    svm_predictions = svm_model.predict(X_test_tfidf)
    
    print("\n--- SVM Results ---")
    print(f"Accuracy: {accuracy_score(y_test, svm_predictions):.4f}")
    print(classification_report(y_test, svm_predictions, target_names=['FAKE (0)', 'REAL (1)']))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.pkl"))
    print("Saved SVM Model.")
    
    print("\nModel training and saving complete!")

if __name__ == "__main__":
    main()

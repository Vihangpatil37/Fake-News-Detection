# 📰 TruthGuard: Fake News Detection System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)](https://scikit-learn.org/)

**TruthGuard** is an advanced, end-to-end Machine Learning web application that analyzes news articles and headlines to predict whether they are **REAL** or **FAKE**. 

Built with a sleek, responsive "glassmorphism" UI using Bootstrap 5, this project showcases modern Natural Language Processing (NLP) techniques and robust classification algorithms.

## ✨ Features
- **Dual AI Engines**: Toggle between a **Support Vector Machine (SVM)** and a **Random Forest Classifier** to compare prediction results.
- **Premium UI/UX**: An animated, modern web interface making fake news detection accessible and visually pleasing.
- **Automated Data Pipeline**: Built-in Python training script that automatically downloads a dataset, preprocesses text with RegEx, and trains models using `TfidfVectorizer`.
- **Confidence Scoring**: Displays the AI's percentage certainty for each prediction.

## 🛠️ Technology Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3, Bootstrap 5, FontAwesome, Jinja2

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vihangpatil37/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note: Ensure you have `scikit-learn`, `flask`, `pandas`, and `numpy` installed successfully).*

## 🧠 Training the Models
If you want to re-train the models from scratch (or if you haven't cloned the `.pkl` models):
```bash
python train_model.py
```
This script will automatically:
1. Download the public `fake_or_real_news.csv` dataset.
2. Clean the text data (remove punctuation, stop words).
3. Extract TF-IDF features.
4. Train & evaluate the Random Forest and SVM models.
5. Save the artifacts (`rf_model.pkl`, `svm_model.pkl`, `tfidf_vectorizer.pkl`) into the `model/` directory.

## 🌐 Running the Web App

Start the Flask development server:
```bash
python app.py
```
Open your web browser and navigate to: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

Paste any article or suspicious headline into the text box and click **Analyze Authenticity**!

---
*Disclaimer: This project is for educational purposes. Machine learning models are not 100% accurate and can inherit bias from their training datasets.*

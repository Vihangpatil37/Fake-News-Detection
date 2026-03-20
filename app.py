import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load models and vectorizer at startup
MODEL_DIR = "model"
# We will use the SVM model as it typically performs very well for text classification and is faster to inference, but user can change it.
# Let's say we prefer SVM but we will load both and pick SVM by default.
try:
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return render_template("index.html", error="Models are not loaded. Please train the models first.")
    
    text = request.form.get("news_text", "").strip()
    model_choice = request.form.get("model_choice", "svm")
    
    if not text:
        return render_template("index.html", error="Please enter some text to analyze.")
    
    # Preprocess and vectorize
    text_tfidf = vectorizer.transform([text])
    
    # Select model
    if model_choice == "rf":
        model = rf_model
        model_name = "Random Forest Classifier"
    else:
        model = svm_model
        model_name = "Support Vector Machine (SVM)"
    
    # Predict
    prediction_num = model.predict(text_tfidf)[0]
    prediction = "REAL" if prediction_num == 1 else "FAKE"
    
    # Get confidence if available
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities) * 100
        
    return render_template("result.html", 
                           prediction=prediction, 
                           confidence=f"{confidence:.2f}%", 
                           model_name=model_name, 
                           original_text=text)

if __name__ == "__main__":
    app.run(debug=True)

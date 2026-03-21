import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load models and vectorizer at startup
MODEL_DIR = "model"
GNEWS_API_KEY = "03dba3d2cf6f3fc1c03e8b80abac9d4e" # Use the provided key directly

try:
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

def check_live_news(query):
    """
    Search for the query on GNews API to see if it exists in recent reputable news.
    Returns: (found_count, top_article_title, top_article_url)
    """
    if not GNEWS_API_KEY:
        return 0, None, None
    
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=3&apikey={GNEWS_API_KEY}"
    try:
        import requests
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if articles:
                return len(articles), articles[0]["title"], articles[0]["url"]
    except Exception as e:
        print(f"GNews API error: {e}")
    
    return 0, None, None


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
    
    # Get confidence
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities) * 100
    
    # Live Verification (GNews)
    # Use only the first 100 characters or the first sentence for a better search
    search_query = text.split('.')[0][:100] 
    live_count, live_title, live_url = check_live_news(search_query)
    
    # Logic: Show verification status, but don't overwrite ML prediction
    verification_status = "Unverified"
    if live_count > 0:
        verification_status = "Verified by Live News"
        # Removed the logic that upgrades FAKE to REAL based on GNews
        # so that it "still shows fake news" as requested.
            
    return render_template("result.html", 
                           prediction=prediction, 
                           confidence=f"{confidence:.2f}%", 
                           model_name=model_name, 
                           original_text=text,
                           verification_status=verification_status,
                           live_url=live_url,
                           live_title=live_title)


if __name__ == "__main__":
    app.run(debug=True)

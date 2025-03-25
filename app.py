from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form["review"]
    
    if not data:
        return jsonify({"error": "No review provided"}), 400
    
    review_vec = vectorizer.transform([data])
    prediction = model.predict(review_vec)[0]
    
    return render_template("index.html", sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)
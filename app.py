import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model, vectorizer, and label encoder
svm_model = joblib.load("svm_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review_vec = vectorizer.transform([review])  # Transform text input
        prediction_encoded = svm_model.predict(review_vec)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]  # Decode label
        return render_template('index.html', review=review, prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)

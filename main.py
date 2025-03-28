import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

# Load dataset
df = pd.read_csv("amazon.csv")

# Drop missing values
df.dropna(subset=['reviewText'], inplace=True)

# Preprocess text
df['cleaned_text'] = df['reviewText'].apply(preprocess_text)

# Convert ratings into sentiment labels
df['sentiment'] = df['overall'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))

# Splitting dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), min_df=3)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train SVM Classification model
svm_classifier = SVC(kernel='linear', class_weight='balanced', probability=True)
svm_classifier.fit(X_train_vec, y_train_encoded)

# Save trained model and vectorizer
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluate the model
y_pred_encoded = svm_classifier.predict(X_test_vec)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"SVM Classification Model - Accuracy: {accuracy:.4f}")

# Extract important words
feature_names = np.array(vectorizer.get_feature_names_out())
coef = svm_classifier.coef_.toarray()
positive_words = feature_names[np.argsort(coef[2])[-10:]]  # Top 10 positive words
negative_words = feature_names[np.argsort(coef[0])[:10]]  # Top 10 negative words

print("Top Positive Words:", positive_words)
print("Top Negative Words:", negative_words)


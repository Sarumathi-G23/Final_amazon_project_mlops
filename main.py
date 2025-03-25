import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("amazon.csv")

# Drop missing values
df.dropna(subset=['reviewText'], inplace=True)

# Convert ratings into sentiment labels
df['sentiment'] = df['overall'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))

# Check class distribution and balance it
print("Class Distribution Before Balancing:\n", df['sentiment'].value_counts())

# Splitting dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df['reviewText'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# Text vectorization with improved settings
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2), min_df=5)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training with class balancing
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
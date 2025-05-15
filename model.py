import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
nltk.download('punkt')

def train_emotion_model():
    df = pd.read_csv('data/journal_data.csv')  # Your dataset
    texts = df['entry']
    labels = df['emotion']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression()
    clf.fit(X, labels)

    joblib.dump(clf, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def predict_emotion(text):
    clf = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X = vectorizer.transform([text])
    return clf.predict(X)[0]

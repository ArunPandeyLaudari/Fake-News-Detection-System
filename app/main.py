# app/main.py

import streamlit as st
import joblib
import os
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.text_cleaner import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.markdown("Paste a news article below to check if it's **fake** or **real**.")

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    # its inside the models directory
    model_path = os.path.join(parent_dir, 'models', 'logistic_regression_model.pkl')
    vectorizer_path = os.path.join(parent_dir, 'models', 'tfidf_vectorizer.pkl')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Input field
user_input = st.text_area("Enter news text here:", height=300, placeholder="Paste the news article...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        cleaned_text = clean_text(user_input)
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max()

        if prediction == 0:
            st.error(f"🚨 Prediction: **Fake News**")
        else:
            st.success(f"✅ Prediction: **Real News**")

        st.write(f"**Confidence:** {confidence * 100:.2f}%")
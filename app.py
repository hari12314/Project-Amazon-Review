import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load dependencies
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Load vectorizer and model
@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = joblib.load("best_tfidf_vectorizer.joblib")
    model = joblib.load("best_model_LogisticRegression.joblib")   # change name if different
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

# Text cleaning
def clean_text(text):
    text = str(text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# Streamlit Pages
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

page = st.sidebar.radio("Navigation", ["Introduction", "Predict Sentiment"])

# PAGE 1: INTRO
if page == "Introduction":
    st.title("Amazon Reviews Sentiment Analysis")
    st.markdown("""
    ### Overview
    This app analyzes **Amazon product reviews** and predicts whether the sentiment is **Positive** or **Negative**.

    #### How It Works
    1. Text is cleaned (stopwords, URLs, punctuation removed)
    2. TF-IDF vectorizer converts text to numeric features
    3. A trained ML model (e.g., Logistic Regression / Naive Bayes) predicts sentiment

    #### Pages
    - **Introduction** – About the project  
    - **Predict Sentiment** – Try your own review text!

    #### Model Info
    - Vectorizer: TF-IDF (max_features=5000, bigrams)
    - Algorithms tried: Logistic Regression, Naive Bayes, LightGBM
    - Evaluation metrics: Accuracy, Precision, Recall, F1-score

    ---
    *Try the Predict page to see it in action!*
    """)

# PAGE 2: PREDICT
elif page == "Predict Sentiment":
    st.title("Predict Sentiment from Text")
    st.write("Enter a review below and click **Analyze** to predict sentiment.")

    user_input = st.text_area("Enter review text:", height=200)

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            clean_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([clean_input])
            pred = model.predict(vectorized_input)[0]

            label = "Positive" if pred == 1 else "Negative"
            st.subheader("Result:")
            st.success(f"The predicted sentiment is **{label}**")

            st.markdown("---")
            st.markdown("### Cleaned Text Preview")
            st.write(clean_input)

# sentiment_analysis_app.py

import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Tweets.csv")  # Ensure the CSV is uploaded
    df = df[['text', 'airline_sentiment']]  # Use only necessary columns
    df.dropna(inplace=True)
    df['clean_text'] = df['text'].apply(clean_text)
    return df

st.title("✈️ Airline Tweet Sentiment Analyzer")
df = load_data()

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])
y = df['airline_sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation (optional)
with st.expander("See model performance"):
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))

# Streamlit user input
user_input = st.text_area("Enter a tweet to analyze sentiment", "The flight was amazing!")

if st.button("Analyze"):
    cleaned_input = clean_text(user_input)
    vectorized_input = tfidf.transform([cleaned_input])
    prediction = model.predict(vectorized_input)[0]
    st.subheader(f"Predicted Sentiment: `{prediction}`")


import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Download NLTK data (first run only)
nltk.download('punkt')
nltk.download('stopwords')

@st.cache(allow_output_mutation=True)
def load_pipeline():
    # Load and preprocess
    df = pd.read_csv('Tweets.csv')
    sw = set(stopwords.words('english'))
    def clean(t):
        t = t.lower()
        t = re.sub(r'http\S+|www\.\S+', '', t)
        t = re.sub(r'[^a-z\s]', '', t)
        tokens = word_tokenize(t)
        return ' '.join(tok for tok in tokens if tok not in sw)
    df['clean'] = df['text'].apply(clean)

    # Vectorize & train
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(df['clean'])
    y = df['airline_sentiment'].map({'positive':1,'neutral':0,'negative':-1})

    nb = MultinomialNB().fit(X, y)
    lr = LogisticRegression(max_iter=200).fit(X, y)
    return vec, nb, lr

vectorizer, nb_model, lr_model = load_pipeline()

st.title("🕊️ Tweet Sentiment Analyzer")
tweet = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    # Clean input
    sw = set(stopwords.words('english'))
    def clean_input(t):
        t = t.lower()
        t = re.sub(r'http\S+|www\.\S+', '', t)
        t = re.sub(r'[^a-z\s]', '', t)
        tokens = word_tokenize(t)
        return ' '.join(tok for tok in tokens if tok not in sw)

    cleaned = clean_input(tweet)
    vec = vectorizer.transform([cleaned])
    nb_pred = nb_model.predict(vec)[0]
    lr_pred = lr_model.predict(vec)[0]
    label = {1:"Positive", 0:"Neutral", -1:"Negative"}

    st.write("**Naive Bayes says:**", label[nb_pred])
    st.write("**Logistic Regression says:**", label[lr_pred])

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Download NLTK data (only once)
nltk.download('punkt')
nltk.download('stopwords')

# 1. Load dataset
df = pd.read_csv('Tweets.csv')  # replace if your file is named differently

# 2. Clean text
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)     # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)             # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

# Map sentiment to numeric labels
label_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
df['label'] = df['airline_sentiment'].map(label_mapping)

# 3. Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train models
nb = MultinomialNB()
nb.fit(X_train, y_train)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# 6. Evaluate
for name, model in [('Naive Bayes', nb), ('Logistic Regression', lr)]:
    preds = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print()

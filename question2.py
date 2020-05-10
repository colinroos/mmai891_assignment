"""
Colin, Roos
10095681
MMAI
2020
891 - Natural Language Processing
May 9, 2020


Submission to Question 2, Part 1
"""

import pandas as pd
import numpy as np
import spacy
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# import spaCy tokenizer (downloaded using $ python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')
punctuation = string.punctuation


def spacy_tokenizer(sentence):
    # Tokenize sentence using spaCy
    tokens = nlp(sentence)

    # Lemmatize tokens
    tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens]

    # Remove punctuation
    tokens = [word for word in tokens if word not in punctuation]

    return tokens


def clean_text(text):
    return text.strip().lower()


class Predictor(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


df_train = pd.read_csv("data/sentiment_train.csv")

# print(df_train.info())
# print(df_train.head())

X_train = df_train['Sentence']
y_train = df_train['Polarity']

df_test = pd.read_csv("data/sentiment_test.csv")

# print(df_test.info())
# print(df_test.head())

X_test = df_test['Sentence']
y_test = df_test['Polarity']

bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

# classifier = LogisticRegression()
# classifier = RandomForestClassifier()
classifier = MLPClassifier(hidden_layer_sizes=(50, 50), solver='adam', random_state=42, early_stopping=True,
                           verbose=True)

pipe = Pipeline([('cleaner', Predictor()),
                ('vectorizer', bow_vector),
                ('classifier', classifier)])

pipe.fit(X_train, y_train)

df_test['Pred'] = pipe.predict(X_test)

# Print f1 score for a threshold of 0
f1 = f1_score(y_test, df_test['Pred'])
accuracy = accuracy_score(y_test, df_test['Pred'])
print(f'f1 score @ threshold of 0: {f1:.3f}')
print(f'accuracy @ threshold of 0: {accuracy:.3f}')

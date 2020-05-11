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
import re
import unidecode
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import autokeras as ak
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model

# import spaCy tokenizer (downloaded using $ python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')
punctuation = string.punctuation
custom_punctuation = ['', '..', '...', '....', '.....', '......']


def spacy_tokenizer(sentence):
    """
    Tokenizer and pre-processing using spaCy
    :param sentence: input sentence
    :type sentence: string
    :return:
    """

    # Remove digits
    sentence = re.sub(r'\d+', '', sentence)

    # Remove unicode characters
    sentence = unidecode.unidecode(sentence)

    # Tokenize sentence using spaCy
    tokens = nlp(sentence)

    # Remove punctuation
    tokens = [word for word in tokens if word.text not in punctuation and word.text not in custom_punctuation]

    # Lemmatize tokens
    tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens]

    return ' '.join(tokens)


# Load Training data
df_train = pd.read_csv("data/sentiment_train.csv")

# Separate labels and features
X_train = df_train['Sentence']
y_train = df_train['Polarity']

# Load Testing data
df_test = pd.read_csv("data/sentiment_test.csv")

# Separate labels and features
X_test = df_test['Sentence']
y_test = df_test['Polarity']

# Instantiate classifier object
classifier = ak.TextClassifier(max_trials=30, seed=42)

# Clean up sentence data using custom tokenizer, convert datatypes for autokeras
X_train_clean = np.array(X_train.apply(spacy_tokenizer), dtype=np.str)
X_test_clean = np.array(X_test.apply(spacy_tokenizer), dtype=np.str)

# Convert datatypes for compatibility with autokeras
y_train_clean = np.array(y_train)
y_test_clean = np.array(y_test)

# Fit the autokeras classifier
classifier.fit(X_train_clean, y_train_clean, epochs=5)

# Extract the best model from search function.
# Note that due to some bugs in autokeras, the best model needs to be extracted by pausing execution using debug mode
# and recording the model layers and hyperparameters.
model = classifier.tuner.get_best_model()
hyp = classifier.tuner.get_best_hyperparameters()

# Make some predictions using the best classifier
df_test['Pred'] = classifier.predict(X_test_clean)

# Print f1 score for a threshold of 0
f1 = f1_score(y_test, df_test['Pred'])
accuracy = accuracy_score(y_test, df_test['Pred'])
print(f'f1 score @ threshold of 0: {f1:.3f}')
print(f'accuracy @ threshold of 0: {accuracy:.3f}')

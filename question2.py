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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils_tokenizer import spacy_tokenizer_string

# Load training data
df_train = pd.read_csv("data/sentiment_train.csv")

# Separate labels from features
X_train = df_train['Sentence']
y_train = df_train['Polarity']

# Load testing data
df_test = pd.read_csv("data/sentiment_test.csv")

# Separate labels from features
X_test = df_test['Sentence']
y_test = df_test['Polarity']

# Initialize vectorizers
vectorizers = [CountVectorizer(tokenizer=spacy_tokenizer_string, ngram_range=(1, 3)),
               TfidfVectorizer(tokenizer=spacy_tokenizer_string, max_df=0.5, min_df=0.05, max_features=2000,
                               ngram_range=(1, 3))]

# Initialize classifiers
classifiers = [LogisticRegression(random_state=42, n_jobs=-1),
               RandomForestClassifier(random_state=42, n_jobs=-1),
               KNeighborsClassifier(n_jobs=-1),
               AdaBoostClassifier(random_state=42),
               GradientBoostingClassifier(random_state=42),
               MLPClassifier(hidden_layer_sizes=(200, 200, 100, 50), solver='adam', random_state=42,
                             early_stopping=True, alpha=0.01),
               BernoulliNB(),
               GaussianNB()]

# Run all classifiers and report results
for classifier in classifiers:
    for vectorizer in vectorizers:
        # Instantiate the pipeline
        pipe = Pipeline([('vectorizer', vectorizer),
                         ('classifier', classifier)])

        # Fit the pipeline
        pipe.fit(X_train, y_train)

        # Make some predictions on the test set
        df_test['Pred'] = pipe.predict(X_test)

        # Print f1 score for a threshold of 0
        f1 = f1_score(y_test, df_test['Pred'])
        accuracy = accuracy_score(y_test, df_test['Pred'])
        print(type(classifier))
        print(type(vectorizer))
        print(f'f1 score @ threshold of 0: {f1:.3f}')
        print(f'accuracy @ threshold of 0: {accuracy:.3f}\n')

# Export rows 5 examples of incorrect answers
# Setup best model
pipe = Pipeline([('vectorizer', vectorizers[0]),
                 ('classifier', classifiers[5])])

# Fit best model
pipe.fit(X_train, y_train)

# Make predictions
df_test['Pred'] = pipe.predict(X_test)

# Save a dataframe that contains the incorrect predictions
df_test[df_test['Polarity'] != df_test['Pred']].to_csv('incorrect_classification_q2.csv')

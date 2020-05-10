"""
Colin, Roos
10095681
MMAI
2020
891 - Natural Language Processing
May 9, 2020


Submission to Question 1, Part 1
"""

import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

df_train = pd.read_csv("data/sentiment_train.csv")

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv("data/sentiment_test.csv")

print(df_test.info())
print(df_test.head())

# Add a column to the DataFrame to store predicted sentiment polarity
df_test['Pred'] = 0

# import spaCy tokenizer (downloaded using $ python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')

# initialize vader (lexicon files downloaded using nltk.download())
sia = SentimentIntensityAnalyzer()

# Calculate sentiment for each sentence in test set
for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
    # Tokenize sentence using spaCy
    sentence = nlp(row.Sentence)

    # Get polarity score from vader
    scores = sia.polarity_scores(sentence.text)

    # Add sentence sentiment to DataFrame
    df_test.iloc[idx, 2] = scores['compound']

# Initialize variables for testing multiple thresholds
f1_scores = []
accuracies = []
thresholds = np.linspace(-1, 1, 50)

for threshold in thresholds:
    # Round polarity score using compounded measure
    df_test.loc[df_test['Pred'] >= threshold, 'Pred_polarity'] = 1
    df_test.loc[df_test['Pred'] < threshold, 'Pred_polarity'] = 0

    # Calculate f1 score for the threshold
    f1_scores.append(f1_score(df_test['Polarity'], df_test['Pred_polarity']))
    accuracies.append(accuracy_score(df_test['Polarity'], df_test['Pred_polarity']))

# Round polarity score using compounded measure
df_test.loc[df_test['Pred'] >= 0, 'Pred_polarity'] = 1
df_test.loc[df_test['Pred'] < 0, 'Pred_polarity'] = 0

print(df_test[df_test['Polarity'] != df_test['Pred_polarity']].head(5))

# Print f1 score for a threshold of 0
f1 = f1_score(df_test['Polarity'], df_test['Pred_polarity'])
accuracy = accuracy_score(df_test['Polarity'], df_test['Pred_polarity'])
print(f'f1 score @ threshold of 0: {f1:.3f}')
print(f'accuracy @ threshold of 0: {accuracy:.3f}')

plt.plot(thresholds, f1_scores)
plt.plot(thresholds, accuracies, color='r')
plt.legend(['F1 Score', 'Accuracy'], loc='lower left')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('F1 and Accuracy Score by Threshold Value')
plt.show()

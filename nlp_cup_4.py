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
from utils_tokenizer import spacy_tokenizer
from datetime import datetime
from collections import Counter


def SentimentAnalyzerEvaluate(X):
    # Add a column to the DataFrame to store predicted sentiment polarity
    X['Pred'] = 0
    X['neg'] = 0
    X['neu'] = 0
    X['pos'] = 0
    X['Pred_polarity'] = 0

    # import spaCy tokenizer (downloaded using $ python -m spacy download en_core_web_sm)
    nlp = spacy.load('en_core_web_sm')

    # initialize vader (lexicon files downloaded using nltk.download('vader_lexicon))
    sia = SentimentIntensityAnalyzer()

    # Calculate sentiment for each sentence in test set
    for idx, row in tqdm(X.iterrows(), total=X.shape[0]):
        # Tokenize sentence using spaCy

        # Get polarity score from vader
        scores = sia.polarity_scores(row['text'])

        # Add sentence sentiment to DataFrame
        X.iloc[idx, 3] = scores['compound']
        X.iloc[idx, 4] = scores['neg']
        X.iloc[idx, 5] = scores['neu']
        X.iloc[idx, 6] = scores['pos']

        if scores['neu'] >= 0.95:
            polarity = 1
        else:
            if scores['compound'] >= 0.3:
                polarity = 1
            else:
                polarity = 0

        X.iloc[idx, 7] = polarity

        # if idx > 500:
        #     break

    # Initialize variables for testing multiple thresholds
    f1_scores = []
    accuracies = []
    thresholds = np.linspace(-1, 1, 50)

    # for threshold in thresholds:
    #     # Round polarity score using compounded measure
    #     X.loc[X['Pred'] >= threshold, 'Pred_polarity'] = 1
    #     X.loc[X['Pred'] < threshold, 'Pred_polarity'] = 0
    #
    #     # Calculate f1 score for the threshold
    #     f1_scores.append(f1_score(X['sentiment'], X['Pred_polarity']))
    #     accuracies.append(accuracy_score(X['sentiment'], X['Pred_polarity']))

    # Round polarity score using compounded measure
    # X.loc[X['Pred'] >= 0.3, 'Pred_polarity'] = 1
    # X.loc[X['Pred'] < 0.3, 'Pred_polarity'] = 0

    # X[X['sentiment'] != X['Pred_polarity']].to_csv('incorrect_classification_nlp.csv')

    # Print f1 score for a threshold of 0
    f1 = f1_score(X['sentiment'], X['Pred_polarity'])
    accuracy = accuracy_score(X['sentiment'], X['Pred_polarity'])
    print(f'f1 score @ threshold of 0: {f1:.3f}')
    print(f'accuracy @ threshold of 0: {accuracy:.3f}')

    # plt.plot(thresholds, f1_scores)
    # plt.plot(thresholds, accuracies, color='r')
    # plt.legend(['F1 Score', 'Accuracy'], loc='lower left')
    # plt.xlabel('Threshold')
    # plt.ylabel('Score')
    # plt.title('F1 and Accuracy Score by Threshold Value')
    # plt.show()

    return X


def SentimentAnalyzerPredict(X):
    # Add a column to the DataFrame to store predicted sentiment polarity
    X['Pred'] = 0

    # import spaCy tokenizer (downloaded using $ python -m spacy download en_core_web_sm)
    nlp = spacy.load('en_core_web_sm')

    # initialize vader (lexicon files downloaded using nltk.download('vader_lexicon))
    sia = SentimentIntensityAnalyzer()

    # Calculate sentiment for each sentence in test set
    for idx, row in tqdm(X.iterrows(), total=X.shape[0]):
        # Tokenize sentence using spaCy
        tokens = spacy_tokenizer(row['text'])

        # Get polarity score from vader
        scores = sia.polarity_scores(' '.join(tokens))

        # Add sentence sentiment to DataFrame
        X.iloc[idx, 2] = scores['compound']

    # Round polarity score using compounded measure
    X.loc[X['Pred'] >= 0, 'Pred_polarity'] = 1
    X.loc[X['Pred'] < 0, 'Pred_polarity'] = 0
    X['Pred_polarity'] = X['Pred_polarity'].astype(int)

    return X


def get_most_common(X, count=50):
    words = []
    for idx, row in tqdm(X.iterrows(), total=X.shape[0]):
        words.extend(spacy_tokenizer(row['text']))

    word_freq = Counter(words)

    return word_freq.most_common(count)


# # Load training data
df_train = pd.read_csv("data/nlp-cup-event-4/sentiment_train.csv")

ret_train = SentimentAnalyzerEvaluate(df_train)

# # Separate labels from features
# print(df_train.info())
# print(df_train.head())

# Load testing data
# df_test = pd.read_csv("data/nlp-cup-event-4/sentiment_test.csv")
#
# # most_common_words = get_most_common(df_test)
# # print(most_common_words)
#
# # Separate labels from features
# print(df_test.info())
# print(df_test.head())

# ret = SentimentAnalyzerPredict(df_train)

# ret_test = SentimentAnalyzerPredict(df_test)
# ret_test.drop(columns=['text', 'Pred'], inplace=True)
# ret_test.rename(columns={'id': 'Id', 'Pred_polarity': 'Predicted'}, inplace=True)
# ret_test.set_index('Id', inplace=True)
# date = datetime.now().strftime('%Y%m%d_%H%M%S')
# ret_test.to_csv(f'nlp_cup_{date}.csv')


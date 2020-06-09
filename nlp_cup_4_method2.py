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
from tqdm import tqdm
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from symspellpy import SymSpell
import pkg_resources

def vader_sentiment_predict(X):
    # Add a column to the DataFrame to store predicted sentiment polarity
    X['Pred'] = 0
    X['Pred_polarity'] = 0

    analyzer = SentimentIntensityAnalyzer()
    sym_spell = SymSpell(2, 7)

    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")

    if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                     count_index=1):
        print("Dictionary file not found")
        return
    if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0,
                                            count_index=2):
        print("Bigram dictionary file not found")
        return

    for idx, row in tqdm(X.iterrows(), total=X.shape[0]):
        text = sym_spell.lookup_compound(row['text'], 2)
        scores = analyzer.polarity_scores(text[0].term)
        # scores = analyzer.polarity_scores(row['text'])

        X.iloc[idx, 2] = scores['compound']

        if scores['neu'] >= 0.95:
            polarity = 1
        else:
            if scores['compound'] >= 0.3:
                polarity = 1
            else:
                polarity = 0

        X.iloc[idx, 3] = polarity

    # Round polarity score using compounded measure
    # X.loc[X['Pred'] >= 0.2, 'Pred_polarity'] = 1
    # X.loc[X['Pred'] < 0.2, 'Pred_polarity'] = 0
    X['Pred_polarity'] = X['Pred_polarity'].astype(int)

    return X


# Load testing data
df_test = pd.read_csv("data/nlp-cup-event-4/sentiment_test.csv")

# Separate labels from features
print(df_test.info())
print(df_test.head())

# ret = SentimentAnalyzerPredict(df_train)

ret_test = vader_sentiment_predict(df_test)
ret_test.drop(columns=['text', 'Pred'], inplace=True)
ret_test.rename(columns={'id': 'Id', 'Pred_polarity': 'Predicted'}, inplace=True)
ret_test.set_index('Id', inplace=True)
date = datetime.now().strftime('%Y%m%d_%H%M%S')
ret_test.to_csv(f'nlp_cup_method2_{date}.csv')


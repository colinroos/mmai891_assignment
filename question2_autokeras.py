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
from sklearn.metrics import f1_score, accuracy_score
import autokeras as ak
from utils_tokenizer import spacy_tokenizer_string

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
X_train_clean = np.array(X_train.apply(spacy_tokenizer_string), dtype=np.str)
X_test_clean = np.array(X_test.apply(spacy_tokenizer_string), dtype=np.str)

# Convert datatypes for compatibility with autokeras
y_train_clean = np.array(y_train)
y_test_clean = np.array(y_test)

# Fit the autokeras classifier
classifier.fit(X_train_clean, y_train_clean, epochs=10)

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

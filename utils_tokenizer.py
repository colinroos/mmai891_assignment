import spacy
import re
import unidecode
import string

# import spaCy tokenizer (downloaded using $ python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')
punctuation = string.punctuation
custom_punctuation = ['', '..', '...', '....', '.....', '......']


def spacy_tokenizer(sentence):
    """
    Tokenizer and pre-processing using spaCy
    :param sentence: input sentence
    :type sentence: string
    :return: list of tokens
    """

    # Remove digits
    sentence = re.sub(r'\d+', '', sentence)

    # Remove unicode characters
    sentence = unidecode.unidecode(sentence)

    # Tokenize sentence using spaCy
    tokens = nlp(sentence)

    # Remove punctuation
    tokens = [word for word in tokens if word.text not in punctuation and word.text not in custom_punctuation]

    # Remove stop words, DON'T USE, not beneficial in this use case
    # tokens = [word for word in tokens if word.text not in STOP_WORDS]

    # Lemmatize tokens
    tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens]

    return tokens

def spacy_tokenizer_string(sentence):
    """
    Tokenizer and pre-processing using spaCy
    :param sentence: input sentence
    :type sentence: string
    :return: string of joined tokens
    """

    # Remove digits
    sentence = re.sub(r'\d+', '', sentence)

    # Remove unicode characters
    sentence = unidecode.unidecode(sentence)

    # Tokenize sentence using spaCy
    tokens = nlp(sentence)

    # Remove punctuation
    tokens = [word for word in tokens if word.text not in punctuation and word.text not in custom_punctuation]

    # Remove stop words, DON'T USE, not beneficial in this use case
    # tokens = [word for word in tokens if word.text not in STOP_WORDS]

    # Lemmatize tokens
    tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens]

    return ' '.join(tokens)

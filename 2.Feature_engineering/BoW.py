import pandas as pd
import numpy as np
import os.path
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
import nltk
import spacy


# create function to clean the text
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """

    # transforms all to lower case words
    mess = mess.lower()

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word not in stopwords.words('english')]

# create a WordNet tokenizer
wrd_lemmatizer = WordNetLemmatizer()

def wn_tokenizer(doc):
    tokens = word_tokenize(doc)
    return wrd_lemmatizer.lemmatize(tokens, pos='v')

def bow_encoding(s):
    '''
    Takes a pandas Series and encodes the strings to a Bag of Words matrix
    :param s: pandas series
    :return: Numpy Matrix
    '''

    BoW_WN = CountVectorizer(analyzer=text_process, tokenizer=wn_tokenizer, min_df=4)
    BoW_WN_matrix = BoW_WN.fit_transform(s)
    with open('CountVectorizer.pk', 'wb') as fin:
        pickle.dump(BoW_WN, fin)
    return BoW_WN_matrix


def tfidf_encoding(s):
    '''
    Takes a series of Texts and encodes the strings to a TFIDF matrix
    :param s: pandas series
    :return: numpy matrix
    '''

    tfidf_vectorizer_WN = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=wn_tokenizer, min_df=4)
    tfidf_WN_matrix = tfidf_vectorizer_WN.fit_transform(s)
    with open('TFIDFVectorizer.pk', 'wb') as fin:
        pickle.dump(tfidf_vectorizer_WN, fin)
    return tfidf_WN_matrix
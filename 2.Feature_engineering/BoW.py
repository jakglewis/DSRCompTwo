import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from gensim.models import Word2Vec
import pickle


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
    Takes a pandas Series and fits and transforms the strings to a Bag of Words matrix
    :param s: pandas series
    :return: Numpy Matrix
    '''

    BoW_WN = CountVectorizer(analyzer=text_process, tokenizer=wn_tokenizer, min_df=4)
    BoW_WN_matrix = BoW_WN.fit_transform(s)
    with open('CountVectorizer.pk', 'wb') as fin:
        pickle.dump(BoW_WN, fin)
    return BoW_WN_matrix

def bow_encoding_val(s):
    '''
    Takes a pandas Series and transforms the strings to a Bag of Words matrix
    :param s: pandas series
    :return: Numpy Matrix
    '''

    with open('CountVectorizer.pk', 'rb') as f:
        BoW_WN = pickle.load(f)
    BoW_WN_matrix = BoW_WN.transform(s)

    return BoW_WN_matrix


def tfidf_encoding(s):
    '''
    Takes a series of Texts and fit and transforms the strings to a TFIDF matrix
    :param s: pandas series
    :return: numpy matrix
    '''

    tfidf_vectorizer_WN = TfidfVectorizer(analyzer=text_process, use_idf=True, tokenizer=wn_tokenizer, min_df=4)
    tfidf_WN_matrix = tfidf_vectorizer_WN.fit_transform(s)
    with open('TFIDFVectorizer.pk', 'wb') as fin:
        pickle.dump(tfidf_vectorizer_WN, fin)
    return tfidf_WN_matrix


def tfidf_encoding_val(s):
    '''
    Takes a series of Texts and transforms the strings to a TFIDF matrix
    :param s: pandas series
    :return: Numpy Matrix
    '''

    with open('TFIDFVectorizer.pk', 'rb') as f:
        tfidf_vectorizer_WN = pickle.load(f)
    tfidf_WN_matrix = tfidf_vectorizer_WN.transform(s)

    return tfidf_WN_matrix

def w2v_create(X_train):
    sentences = []
    col = X_train.loc[:,X_train.dtypes == object]
    for i in range(col.shape[1]):
        sentences.extend(col.iloc[:,i].tolist())
    w2v_model = Word2Vec(sentences, min_count=4)
    with open('w2v_model.pk', 'wb') as fin:
        pickle.dump(w2v_model, fin)
    pass

def transform_vocab(wordlist):
    '''
    Use dataframe.apply(transform_vocab) to transform the list of words in each Dataframe column to a list of numpy array vectors
    :param wordlist: list of words
    :return: list of numpy arrays with word vector (length: 100) and padded sequence (length:20)
    '''
    with open('w2v_model.pk', 'rb') as f:
        model = pickle.load(f)
    filtered_wl = [word for word in wordlist if word in model.wv.vocab]
    vectorlist = model.wv[filtered_wl]
    paddings = 20 - len(vectorlist)
    padded_vectorlist = paddings * [100 * [0]] + vectorlist[0:20].tolist()
    return np.asarray(padded_vectorlist)

def w2v_transform(X_train):
    '''
    Transforms the cleaned Dataframe from containing a list of words to containing a padded sequence of word vectors.
    :param X_train:
    :return: padded vectorized Dataframe
    '''
    idx = [X_train.columns.get_loc(c) for c in X_train.filter(like='Top').columns if c in X_train]
    for i in idx:
        X_train.iloc[:, i] = X_train.iloc[:, i].apply(transform_vocab)
    return X_train
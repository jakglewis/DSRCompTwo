#### This file:
# Takes raw training data, recalculates the target from stock price data
# Splits the data into train and val
# Encodes the imput data for 

import pandas as pd
import os.path
import datetime
from pandas import datetime

import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle

import os.path
from os import path

# Data Loading for training data
df = pd.read_csv('0.Raw_data/train/Combined_News_DJIA_train.csv')   # please check if Training data is in the same location on your PC
df = df.set_index("Date")

# Data loading for test data

if path.exists("data/test/Combined_News_DJIA_test.csv"):
    df_test = pd.read_csv('data/test/Combined_News_DJIA_test.csv') 


# read in stock prices data for re-creating target values, given that some have been removed.

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

series = pd.read_csv('0.Raw_data/DJIA_table.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = series.sort_index(ascending=True)

#create new target values_train data
series['lag_adj_close'] = series['Adj Close'].shift(1)
series['diff_close'] = series['Adj Close']-series['lag_adj_close']
series['target'] = 0
series['target'] = (series['diff_close']>=0 ).map(int)
#series['target'] = diff
series['target']
series_new = series.loc[:'2014-11-20']
new_df=pd.concat([series_new['target'],df['Label']],axis=1)

df['Label']  = series_new['target']

# create Y test values from stock data

if path.exists("data/test/DJIA_table_test.csv"):
    y_test = pd.read_csv('data/test/DJIA_table_test.csv') 
    y_test['lag_adj_close'] = y_test['Adj Close'].shift(1)
    y_test['diff_close'] = y_test['Adj Close']-y_test['lag_adj_close']
    y_test['target'] = 0
    y_test['target'] = (y_test['diff_close']>=0 ).map(int)
    Y_test = y_test['target'] 



# Data Cleaning and merging all headlines to one single column 
def clean_and_merge(df):
    if df.shape[1] > 27:
        subdf = df.iloc[:,2:27]
    else:
        subdf = df.iloc[:,1:26]
    subdf = subdf.applymap(str)
    s = subdf.apply(lambda x: ' '.join(x), axis=1)

    # replace the b' and b" which are in the beginning of some headlines
    s = s.str.replace("b'","")
    s = s.str.replace('b"','')

    # create a new dataframe with all headlines and the overall word count
    df1 = s.to_frame(name='headlines')
    df1['lengths'] = df1['headlines'].apply(len)
    df1['date'] = df.index
    
    return df1

df1 = clean_and_merge(df)
if path.exists("data/test/Combined_News_DJIA_test.csv"):
    df2 = clean_and_merge(df_test)

###split data set (train only)

val_size = 0.2
train_length = int(len(df1)*(1-val_size))
train = df1.iloc[:train_length,:]
val = df1.iloc[train_length:,:]


#split dataset base don time
X_train = train.loc[:,]
X_val = val.loc[:,]
y_train = df.iloc[:train_length,:]
y_train=y_train['Label']
y_val = df.iloc[train_length:,:]
y_val=y_val['Label']



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


# Encoding the training data set BoW
df_encode = bow_encoding(X_train['headlines'])
#df_encode

# transforming
#from BoW import bow_encoding_val
df_encode_val = bow_encoding_val(X_val['headlines'])
#df_encode_val

if path.exists("data/test/Combined_News_DJIA_test.csv"):
    # encode test data
    df_encode_test = bow_encoding_val(df2['headlines'])
    #df_encode_test


#
df_encode_TFIDF = tfidf_encoding(X_train['headlines'])
#df_encode_TFIDF


df_encode_TFIDF_val = tfidf_encoding_val(X_val['headlines'])
#df_encode_TFIDF_val

if path.exists("data/test/Combined_News_DJIA_test.csv"):

    df_encode_TFIDF_test = tfidf_encoding_val(df2['headlines'])
    #df_encode_TFIDF_test

X_bow_train= pd.DataFrame(df_encode.toarray())
X_bow_val= pd.DataFrame(df_encode_val.toarray())
X_TFIDF_train= pd.DataFrame(df_encode_TFIDF.toarray())
X_TFIDF_val= pd.DataFrame(df_encode_TFIDF_val.toarray())

if path.exists("data/test/Combined_News_DJIA_test.csv"):
    X_bow_test= pd.DataFrame(df_encode_test.toarray())
    X_TFIDF_test= pd.DataFrame(df_encode_TFIDF_test.toarray())

Y_train = y_train
Y_val = y_val

Y_all = pd.DataFrame(pd.concat([Y_train,Y_val], axis=0))

X_bow_train.to_pickle('4.Data/X_bow_train_v3.pkl')
X_bow_val.to_pickle('4.Data/X_bow_val_v3.pkl')
X_TFIDF_train.to_pickle('4.Data/X_TFIDF_train_v3.pkl')
X_TFIDF_val.to_pickle('4.Data/X_TFIDF_val_v3.pkl')

Y_train.to_pickle('4.Data/Y_train_v3.pkl')
Y_val.to_pickle('4.Data/Y_val_v3.pkl')
Y_all.to_pickle('4.Data/Y_all_v3.pkl')

if path.exists("data/test/Combined_News_DJIA_test.csv"):
    X_bow_test.to_pickle('4.Data/X_bow_test.pkl')
    X_TFIDF_test.to_pickle('4.Data/X_TFIDF_test.pkl')
    Y_test.to_pickle('4.Data/Y_test.pkl')
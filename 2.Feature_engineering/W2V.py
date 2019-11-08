import pandas as pd
import os.path
import datetime
import BoW
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Data Loading
path = os.getcwd()
parent_folder, current_folder = os.path.split(path)
df = pd.read_csv(parent_folder + '/0.Raw_data/train/Combined_News_DJIA_train.csv')   # please check if Training data is in the same location on your PC

def cleaning(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word not in stopwords.words('english')]

# Data Cleaning and merging all headlines to one single column

df.iloc[:,2:27] = df.iloc[:,2:27].applymap(str)

# replace the b' and b" which are in the beginning of some headlines
df.iloc[:,2:27] = df.iloc[:,2:27].replace(regex="b'",value="")
df.iloc[:,2:27] = df.iloc[:,2:27].replace(regex='b"',value='')
df.iloc[:,2:27] = df.iloc[:,2:27].apply(lambda x: x.astype(str).str.lower())

#df1['lengths'] = df1['headlines'].apply(len)
df['Date'] = pd.to_datetime(df['Date'])

for i in range(2,27):
    df.iloc[:,i] = df.iloc[:,i].apply(cleaning)

X_train, X_val, y_train, y_val = train_test_split(df.loc[:, df.columns != 'Label'], df['Label'], test_size=0.20, random_state=2)

sentences = []
for i in range(1, 25):
    sentences.extend(X_train.iloc[:, i].tolist())

len(sentences)


model = Word2Vec(sentences, min_count=2)

def transform_vocab(wordlist):
    filtered_wl = [word for word in wordlist if word in model.wv.vocab]
    return model.wv[filtered_wl]

for i in range(1,26):
    X_train.iloc[:,i] = X_train.iloc[:,i].apply(transform_vocab)

def padding(wordlist):
    paddings = 20 - len(wordlist)
    padded_wordlist = paddings *[100*[0]] + wordlist[0:20].tolist()
    return padded_wordlist

for i in range(1,26):
    X_train.iloc[:,i] = X_train.iloc[:,i].apply(padding)


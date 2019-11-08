
#%% 0. Housekeeping

# =============================================================================
# 0.1 Packages
# =============================================================================

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize   
import gensim 
from gensim.models import Word2Vec 
from tensorflow.keras import preprocessing
import collections
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =============================================================================
# 0.2 Loading in paths
# =============================================================================

data_path = "/home/paul/Documents/1. DSR/1. Competition/DSRCompTwo/0.Raw_data/"
pickle_path = "/home/paul/Documents/1. DSR/1. Competition/DSRCompTwo/Pickles/"

#%% 1. Loading in the data

raw_data = pd.read_csv(data_path + "Combined_News_DJIA.csv")
y_data = raw_data.loc[:,"Label"]

#%% 2. Data Cleaning

# =============================================================================
# 2.1 Housekeeping
# =============================================================================

# Article columns
article_names = raw_data.iloc[:,2:].columns

# Loading in stepwords
stop_words = set(stopwords.words('english')) 

# Creating dataframe for clean text
cleaned_data = pd.DataFrame(data="",
                            columns=article_names,
                            index=data.index)


# =============================================================================
# 4.3 Manual cleaning
# =============================================================================
        
def cleaning(text):
    clean = text.replace('b\'', "")
    clean = clean.replace('\'', "")
    clean = clean.replace('b"', "")
    clean = clean.replace('[VIDEO]', "")
    clean = clean.replace('BREAKING: ', "")
    clean = clean.replace("Donald Trump", "Trump")
    clean = clean.replace('U.S.', "US")
    clean = clean.replace('U.N.', "UN")
    return clean

# =============================================================================
# 4.2 Looping over articles as well as over time
# =============================================================================

# Changing nans into empty strings
raw_data.fillna(" ", inplace=True)
raw_data.set_index("Date", inplace=True)

# Looping over all articles    
for article in article_names: 
    # Looping over all dates
    for date in stop_punc_data.index:
        # Getting the text
        text = raw_data.loc[date,article]
        # Manual cleaning
        text = cleaning(text)
        # Creating empty string holder
        clean_text = []
        # Tokenize the words
        word_tokens = nltk.word_tokenize(text)
        # Looping and removing stopwords and punctuation
        for word in word_tokens:
            # Remove signs
            no_punc = word.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation) ))
            # Remove stopwords
            if no_punc not in stop_words:
                # Appending all text
                clean_text.append(lemmatizer.lemmatize(no_punc.lower()))
                # Assigning the data back to a dataframe
                cleaned_data.loc[date,article] = " ".join(clean_text)

# Saving data
cleaned_data.to_pickle(pickle_path + 'cleaned_data.pkl')

#%% 3. Creating a list of all words used in the dataset

# =============================================================================
# 3.1 Adding all text together
# =============================================================================

# Adding all text together
all_text = ""

# Adding text together to one big text datafile
for article in article_names:
    for date in data.index:
        all_text += " " + cleaned_data.loc[date,article]

list_words = all_text.split()

# =============================================================================
# 3.2 Creating dictionary of words, using the 10.000 most common words
# =============================================================================
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    return dictionary

word_tuple = build_dataset(list_words,vocabulary_size)
word_dictionary = dict(word_tuple)

# =============================================================================
# 3.3 Creating numbers for words
# =============================================================================

# Creating empty copy of text container
num_data = pd.DataFrame(data=[],
                        columns=article_names,
                        index=data.index,
                        )

# Creating a function which changes words to dictionary numbers
def exchange(text, dictionary):    
    # Creating empty list
    number_version = []
    # Tokenize words
    word_tokens = nltk.word_tokenize(text)
    for word in word_tokens:
        if word in dictionary:
            number_version.append(word_dictionary[word])            
        else:
            number_version.append(0)            
    return number_version

# Turning article header to numbers
for article in article_names:
    for date in data.index:    
        text = cleaned_data.loc[date,article] 
        num_data.loc[date,article] = exchange(text, word_dictionary)


#%% 4. Building the model

# =============================================================================
# 4.1 Hyperparameters
# =============================================================================

vocabulary_size = 10000
sequence_lenght = 30
embedding_size = 20

# =============================================================================
# 4.2 Padding articles
# =============================================================================

# Broadcast y variable
y_ext = np.repeat(y_data,25)

# Stack dataframe 
num_stacked = num_data.stack()

# Create a list out of all vectors
num_padded= preprocessing.sequence.pad_sequences(
    num_stacked,
    maxlen=sequence_lenght, 
)

assert len(y_ext) == len(num_stacked)
    
# =============================================================================
# 4.3 Model outline
# =============================================================================

from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Embedding(
    vocabulary_size,
    embedding_size,
    input_length=sequence_lenght,
))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))


model.summary()

#%% 5. Compiling the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

#%% 6. Running the model

history = model.fit(
    num_padded,
    y_ext.values,
    epochs=2,
    batch_size=32,  
    validation_split=0.2,
)

#%% 7. Graphical representation

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.close()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
plt.close()


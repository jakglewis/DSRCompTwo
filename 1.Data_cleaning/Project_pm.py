#%% 0. Paths

data_path = "/home/paul/Documents/1. DSR/1. Competition/DSRCompTwo/0.Raw_data/"

#%% 1. Packages

import pandas as pd
import nltk
import en_core_web_lg # Has to be pip installed
nlp = en_core_web_lg.load()
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string 
import re
import nltk
nltk.download('vader_lexicon')
import operator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


#%% 2. Importing data

data = pd.read_csv(data_path + "Combined_News_DJIA.csv")

#%% 3. Combining news from a day

# Changing nans into empty strings
data.fillna(" ", inplace=True)

# Getting column names only for the articles
article_names = data.iloc[:,2:].columns

#%% 4. Remove stop words and punctuasiation

# =============================================================================
# 4.1 House keeping
# =============================================================================

# Loading in stepwords
stop_words = set(stopwords.words('english')) 

# Creating dataframe for clean text
stop_punc_data = pd.DataFrame(data="",
                              columns=article_names,
                              index=data.index)

# Creating dataframe for clean text
no_b = pd.DataFrame(data="",
                    columns=article_names,
                    index=data.index)

# =============================================================================
# 4.2 Looping over articles as well as over time
# =============================================================================

# Looping over all articles of the news
for article in article_names:    
    # Looping over all dates
    for date in data.index:
        # Getting the text
        text = data.loc[date,article]
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
                clean_text.append(no_punc)
                # Assigning the data back to a dataframe
                stop_punc_data.loc[date,article] = " ".join(clean_text)

# Saving data
stop_punc_data.to_pickle(data_path + 'stop_punc_data.pkl')

# =============================================================================
# 4.3 Manual cleaning of specifics
# =============================================================================

# Removing the b in the beginning
for article in article_names:
    no_b.loc[:,article] = stop_punc_data.loc[:,article].str.replace("b", " ")

# Saving data
no_b.to_pickle(data_path + "no_b.pkl")


#%% 5. Loading the data for quicker usage 
    
stop_punc_data = pd.read_pickle(data_path + "stop_punc_data.pkl")
no_b = pd.read_pickle(data_path + "no_b.pkl")

#%% 6. Getting the top three words using spaCy

top_words = pd.DataFrame(data="",
                         columns=article_names,
                         index=["GPE", "PERSON", "ORG", "Sentiment"])


for article in article_names: 
    
    #for date in data.index:
    date = 0    
    # Storing the text in a variable
    text = no_b.loc[date,article]
    
    # Word recognition of the words in the article to figure what they mean
    doc = nlp(text)
    category = [x.label_ for x in doc.ents]
    Words = [x.lemma_ for x in doc.ents]

    # Grouping words by their frequency
    df = pd.DataFrame({"Category":category, "Words":Words})
    tops = df.groupby('Category')['Words'].apply(lambda x: x.value_counts())
   
# =============================================================================
# 6.1 Assigning top words as well as sentiment
# =============================================================================

    try:
        top_words.loc["GPE",article] = tops.loc[["GPE"]] 
    except:
        top_words.loc["GPE",article] = []
        
    try: 
        top_words.loc["PERSON",article] = tops.loc[["PERSON"]]
    except: 
        top_words.loc["PERSON", article] = []
        
    try:
        top_words.loc["ORG",article] = tops.loc[["ORG"]]
    except: 
        top_words.loc["ORG", article] = []
    
    # Assessing the sentiment
    sentiment = sid.polarity_scores(text)
    positive_score = sentiment["pos"]
    negative_score = -sentiment["neg"]
    
    # Checking which is higher 
    if positive_score >= abs(negative_score):
        top_words.loc["Sentiment", article] = positive_score
    else: 
        top_words.loc["Sentiment", article] = negative_score
  
#%% Word Embedding
    

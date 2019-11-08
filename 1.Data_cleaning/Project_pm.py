#%% 0. Paths

data_path = "/home/paul/Documents/1. DSR/1. Competition/DSRCompTwo/0.Raw_data/"
pickle_path = "/home/paul/Documents/1. DSR/1. Competition/DSRCompTwo/Pickles/"

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

# Changing index
data.set_index("Date", inplace=True)

# Changing nans into empty strings
data.fillna(" ", inplace=True)

# Getting column names only for the articles
article_names = data.iloc[:,2:].columns

# Combining the articles
data.loc[:,"total"] = ""

for article in article_names:
    data.loc[:,"total"] += " " + data.loc[:,article] 

# Removing bs
for date in data.index: 
    data.loc[date,"total"] = data.loc[date,"total"].replace('b\'', "")
    data.loc[date,"total"] = data.loc[date,"total"].replace('\'', "")
    data.loc[date,"total"] = data.loc[date,"total"].replace('b"', "")
    data.loc[date,"total"] = data.loc[date,"total"].replace('[VIDEO]', "")
    data.loc[date,"total"] = data.loc[date,"total"].replace('BREAKING: ', "")

#%% 4. Remove stop words and punctuasiation

# =============================================================================
# 4.1 House keeping
# =============================================================================

# Loading in stepwords
stop_words = set(stopwords.words('english')) 

# Creating dataframe for clean text
stop_punc_data = pd.DataFrame(data="",
                              columns=["total"],
                              index=data.index)

# =============================================================================
# 4.2 Looping over articles as well as over time
# =============================================================================
    
# Looping over all dates
for date in stop_punc_data.index:
    # Getting the text
    text = data.loc[date,"total"]
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
            clean_text.append(no_punc.lower())
            # Assigning the data back to a dataframe
            stop_punc_data.loc[date,"total"] = " ".join(clean_text)

# =============================================================================
# 4.3 Manual cleaning of specifics
# =============================================================================

for date in stop_punc_data.index: 
    stop_punc_data.loc[date,"total"] = stop_punc_data.loc[date,"total"].replace("Donald Trump", "Trump")
    stop_punc_data.loc[date,"total"] = stop_punc_data.loc[date,"total"].replace("the", "")
    stop_punc_data.loc[date,"total"] = stop_punc_data.loc[date,"total"].replace("U S", "US")
    stop_punc_data.loc[date,"total"] = stop_punc_data.loc[date,"total"].replace("U N", "UN")

# Saving data
stop_punc_data.to_pickle(pickle_path + 'stop_punc_data.pkl')

#%% 5. Loading the data for quicker usage 
    
stop_punc_data = pd.read_pickle(pickle_path + "stop_punc_data.pkl")

#%% 6. Getting the top three words using spaCy

categories = ["GPE", "ORG", "PERSON"]
position = ["1"]

combination = [category+number for category in categories for number in position]

top_words = pd.DataFrame(data="",
                         columns=combination,
                         index=data.index)


for date in top_words.index: 
    
    # Storing the text in a variable
    text = stop_punc_data.loc[date,"total"]
    
    # Word recognition of the words in the article to figure what they mean
    doc = nlp(text)
    category = [x.label_ for x in doc.ents]
    Words = [x.lemma_ for x in doc.ents]
    
    # Grouping words by their frequency
    df = pd.DataFrame({"Category":category, "Words":Words})
    tops = df.groupby('Category')['Words'].apply(lambda x: x.value_counts())
    category_keys = tops.loc[categories]
    top3 = category_keys.groupby("Category").head(1)
    
# =============================================================================
# 6.1 Assigning top words as well as sentiment
# =============================================================================
       
    try:
        top_words.loc[date, ["GPE1"]] = top3.loc["GPE"].index.to_list()
    except:
        pass

    try:
        top_words.loc[date, ["PERSON1"]] = top3.loc["PERSON"].index.to_list()
    except:
        pass

    try:
        top_words.loc[date, ["ORG1"]] = top3.loc["ORG"].index.to_list()
    except:
        pass

    # Assessing the sentiment
    sentiment = sid.polarity_scores(text)
    positive_score = sentiment["pos"]
    negative_score = -sentiment["neg"]
    
    # Checking which is higher 
    if positive_score >= abs(negative_score):
        top_words.loc[date, "Sentiment"] = positive_score
    else: 
        top_words.loc[date, "Sentiment"] = negative_score

# =============================================================================
# 6.2 Saving the output    
# =============================================================================

top_words.to_pickle(pickle_path + 'top_words.pkl')

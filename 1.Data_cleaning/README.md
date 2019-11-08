#Stock market prediction from news headlines
Adapted DSR competition from Kaggle competition:
Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. 
Retrieved 2019-08-19 from https://www.kaggle.com/aaron7sun/stocknews

###Target
The task is to predict the daily change in the Dow Jones Industrial Average (DJIA). The task is formulated as a binary classification task:

* 0 when DJIA Adj Close value decreased
* 1 when DJIA Adj Close value rose or stayed as the same

### Data
This dataset was taken from the Kaggle competition [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews/).

The dataset is from two sources:

the r/worldnews Reddit - the top 25 headlines ranked by upvotes
Yahoo Finance


### Run Model
To setup the environment and create the test set, run

```
pip install -r requirements.txt

python data.py --test 1
```

To run  the model: 

```
Create_Train_BOW_TFIDF_Data.py
```
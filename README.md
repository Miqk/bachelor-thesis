# Predicting BTC/USDT BINANCE price using Twitter
Scripts made for bachelor's thesis. Libraries used: NumPy, pandas, requests

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
Predictions are made using both Vader and BERT. Then they are verified for for certain adjustable dates, intervals and lags. You can also set up certain ngrams which you want to opt out of.
Sentiment change threshold is also adjustable.
## Technologies
* Python 3.8
* NumPy 1.20.1
* pandas 1.2.4
* requests 2.25.1
* transformers 4.12.5
* nltk 3.6.1
## Setup
* Generate Twitter data with certain hashtag using snscrape.
* Change the directory in files to where you saved the twitter data.
* Use twitter_data_manipulation.py to lemmatize & tokenize the words and get sentiment predictions.
* Check predictions using prediction_checker.py

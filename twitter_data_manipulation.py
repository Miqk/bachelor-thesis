import ijson
import pandas as pd
import nltk
import string
import re
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
from datetime import datetime

logger = logging.getLogger('tipper')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class TwitterData:
    def __init__(self, file_tuple, language):
        self.file_tuple = file_tuple
        self.language = language
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.bert_classifier = pipeline('sentiment-analysis')
        self.vader_classifier = SentimentIntensityAnalyzer()
        self.twitter_data = self.get_processed_data()

    def files_to_df(self):
        df = pd.DataFrame()
        logger.info(f'{datetime.now()} Starting to concatenate files')
        for file in self.file_tuple:
            with open(file, 'r') as f:
                objects = ijson.items(f, '', multiple_values=True)
                df = df.append(pd.DataFrame(([pd.to_datetime(row['date']), row['content'], row['replyCount'],
                                              row['retweetCount'], row['likeCount'], row['quoteCount'], row['lang']]
                                             for row in objects),
                                            columns=self.get_columns()), ignore_index=True)
        logger.info(f'{datetime.now()} File concatenation done')
        return df[df.lang == self.language].iloc[:, :-1].dropna() if self.language else df.dropna()

    def cleanup_data(self, tweet_content):
        return [self.lemmatizer.lemmatize(word) for word in
                [word for word in
                 re.split('\W+', re.sub('[0-9]+', '',
                                        ''.join([char for char in re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', tweet_content)
                                                 if char not in string.punctuation]))) if word not in self.stopwords]]

    def get_processed_data(self):
        df = self.files_to_df()
        df['processed_content'] = df['content'].apply(lambda x: self.cleanup_data(x))
        logger.info(f'{datetime.now()} Tweet cleanup finished')
        df['bert'] = df['content'].apply(lambda x: self.bert_classifier(x)[0])
        logger.info(f'{datetime.now()} BERT classification finished')
        df['vader'] = df['content'].apply(lambda review: self.vader_classifier.polarity_scores(str(review)))
        logger.info(f'{datetime.now()} VADER classification finished')
        df = pd.concat([df.reset_index(drop=True),
                       pd.DataFrame.from_records(df['bert']).reset_index(drop=True),
                        pd.DataFrame.from_records(df['vader']).reset_index(drop=True)],
                       axis=1).drop(['bert', 'vader'], axis=1)
        return df

    @staticmethod
    def get_columns():
        return ['date', 'content', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'lang']


if __name__ == '__main__':
    dane = TwitterData((r'E:\bachelor\btc_01_07.json', r'E:\bachelor\btc_07_14.json', r'E:\bachelor\btc_14_21.json', r'E:\bachelor\btc_21_31.json'), 'en')
    dane.twitter_data.to_csv(r'E:\bachelor\twitter_data_with_sentiment_values_12_12.json')
    #dane = TwitterData((r'E:\bachelor\text-query-tweets.json', r'E:\bachelor\text-query-tweets1.json'), 'en')




import pandas as pd
import numpy as np
import logging
from datetime import datetime
from price_data_generator import DataGenerator
import itertools
from ast import literal_eval
pd.options.mode.chained_assignment = None

logger = logging.getLogger('tipper')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

TWITTER_DATA_PATH = r'E:/bachelor/twitter_data_with_sentiment_values_12.csv'


class PredictionCheck:
    def __init__(self, threshold, date_start, date_end, intervals, shifts, ngram_list):
        self.threshold, self.date_start, self.date_end, self.intervals = threshold, date_start, date_end, intervals
        self.shifts, self.ngram_list = shifts, ngram_list
        self.twitter_data = self.filter_df_for_ngrams(
            pd.read_csv(TWITTER_DATA_PATH), self.ngram_list)
        self.vader_df = self.twitter_data[abs(self.twitter_data.compound) > 0.5]
        self.bert_df = self.create_bert_df(self.twitter_data)
        self.evaluation = self.merge_scores_for_methods()

    def calculate_interval_predictions(self, interval, df, score_column):
        """Retrieve predictions for certain interval"""
        dg = self.get_adj_prices(interval)
        interval_df = self.get_adj_twitter_df(df, score_column, interval)
        return dg.merge(interval_df, left_index=True, right_index=True)

    def merge_scores_for_methods(self):
        """Merge scores for both VADER and BERT methods into one DataFrame"""
        return pd.concat([self.get_final_scores(self.vader_df, 'compound'),
                          self.get_final_scores(self.bert_df, 'bert_score')])

    @staticmethod
    def create_bert_df(df):
        """Create df for BERT calculations from twitter data"""
        logger.info(f'Creating bert df')
        df['label'] = df['label'].apply(lambda x: 1 if str(x) == 'POSITIVE' else -1)
        df['bert_score'] = df['score'] * df['label']
        return df[abs(df['bert_score']) > 0.5]

    @staticmethod
    def determine_method_col(col_name):
        return 'vader' if col_name == 'compound' else 'bert'

    def get_final_scores(self, df, score_column):
        """Retrieve final scores for both methods, shifts and intervals"""
        logger.info(f'Getting final scores for {self.determine_method_col(score_column)}')
        return pd.DataFrame([list(itertools.chain.from_iterable(
            [[interval, shift, self.determine_method_col(score_column)],
             self.calc_conf_matrix_stats(interval, shift, df, score_column)]
                                  ))
                             for shift in range(self.shifts)
                             for interval in self.intervals], columns=self.get_final_scores_columns())

    @staticmethod
    def get_final_scores_columns():
        return ['interval', 'lag', 'method', 'accuracy', 'precision', 'recall', 'f1_score']

    def get_confusion_matrix(self, interval, lag, df, score_column):
        """Retrieve confusion matrix for certain prediction"""
        logger.info(f'{datetime.now()} Calculating confusion matrix for {self.determine_method_col(score_column)},'
                    f' interval: {interval} and lag: {lag}')
        df = self.calculate_interval_predictions(interval, df, score_column)[['real_pred', 'price_up']]
        df['price_up'] = df['price_up'].shift(lag+1)
        df.drop(df[df['real_pred'] == 0].index, inplace=True)
        df['tp'] = np.where((df['real_pred'] == df['price_up']) & (df['real_pred'] == 1), 1, 0)
        df['tn'] = np.where((df['real_pred'] == df['price_up']) & (df['real_pred'] == -1), 1, 0)
        df['fp'] = np.where((df['real_pred'] == 1) & (df['price_up'] == -1), 1, 0)
        df['fn'] = np.where((df['real_pred'] == -1) & (df['price_up'] == 1), 1, 0)
        return {'tp': np.sum(df['tp']), 'fn': np.sum(df['fn']), 'fp': np.sum(df['fp']), 'tn': np.sum(df['tn'])}

    def calc_conf_matrix_stats(self, interval, lag, df, score_column):
        """Retrieve Accuracy, Precision, Recall and F1 Score"""
        conf_matrix = self.get_confusion_matrix(interval, lag, df, score_column)
        return [self.calc_accuracy(conf_matrix), self.calc_precision(conf_matrix), self.calc_recall(conf_matrix),
                self.calc_f1_score(self.calc_recall(conf_matrix), self.calc_precision(conf_matrix))]

    @staticmethod
    def calc_accuracy(conf_matrix):
        return (conf_matrix['tp'] + conf_matrix['tn']) / sum(conf_matrix.values())

    @staticmethod
    def calc_recall(conf_matrix):
        return conf_matrix['tp'] / (conf_matrix['fn'] + conf_matrix['tp'])

    @staticmethod
    def calc_precision(conf_matrix):
        return conf_matrix['tp'] / (conf_matrix['fp'] + conf_matrix['tp'])

    @staticmethod
    def calc_f1_score(recall, precision):
        return (2 * recall * precision) / (recall + precision)

    @staticmethod
    def merge_dfs_with_shift(prices, twtr, shift, freq):
        """Merge price and twitter DataFrames with certain shift"""
        return twtr.merge(prices.shift(shift, freq=freq), how='inner', left_index=True, right_index=True)

    @staticmethod
    def interval_match(interval):
        """Match intervals (Binance and pandas use different intervals)"""
        return interval[:-2].lower() if 'Min' in interval else interval.lower()

    def get_adj_prices(self, interval):
        """Retrieve price data and whether the price went up"""
        dg = DataGenerator(self.interval_match(interval), self.date_start, self.date_end).data
        dg['price_up'] = np.where(dg.open < dg.close, 1, -1)
        return dg.set_index('date')

    def get_adj_twitter_df(self, df, diff_col, interval):
        """Manipulate twitter data: aggregate tweets for interval, return prediction if they met threshold"""
        df['date'] = pd.to_datetime(df['date'])
        df_twtr = df.groupby(pd.Grouper(key='date', freq=interval)).aggregate(np.mean)
        df_twtr['diff'] = df_twtr[diff_col] - df_twtr[diff_col].shift(1)
        df_twtr['meets_threshold'] = np.where(abs(df_twtr['diff'] / df_twtr[diff_col]) > self.threshold, 1, 0)
        df_twtr['pred'] = np.where(df_twtr['diff'] > 0, 1, -1)
        df_twtr['real_pred'] = df_twtr['meets_threshold'] * df_twtr['pred']
        return df_twtr

    @staticmethod
    def check_for_ngrams(sentence, ngram):
        """Check if row contains ngrams"""
        return 1 if all(sentence.split().__contains__(wrd) for wrd in ngram) else 0

    def filter_df_for_ngrams(self, df, ngrams_list):
        """Discards all rows that contain ngrams"""
        logger.info(f'Discarding unwanted ngrams')
        df = self.word_list_to_str(df)
        df['ngram_count'] = df['processed_str'].apply(lambda sentence: np.sum([self.check_for_ngrams(sentence, ngram)
                                                                               for ngram in ngrams_list]))
        return df[df['ngram_count'] == 0]

    @staticmethod
    def word_list_to_str(df):
        df['processed_content'] = df.processed_content.apply(lambda x: literal_eval(x))
        df['processed_str'] = df['processed_content'].apply(lambda x: ' '.join([str(word).lower() for word in x]))
        return df


if __name__ == '__main__':
    sentiment_change_threshold = 0.05
    start_date = '01/10/21'
    end_date = '01/11/21'
    interval_list = ['15Min', '30Min', '1H', '2H', '4H']
    lag_number = 6
    unwanted_ngrams = [['join', 'astroswap'], ['whale', 'alert']]
    test = PredictionCheck(sentiment_change_threshold, start_date, end_date, interval_list, lag_number, unwanted_ngrams).evaluation
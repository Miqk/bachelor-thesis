import pandas as pd
import numpy as np
from price_data_generator import DataGenerator
pd.options.mode.chained_assignment = None


class PredictionCheck:
    def __init__(self, threshold, date_start, date_end, intervals, shifts):
        self.threshold, self.date_start, self.date_end, self.intervals = threshold, date_start, date_end, intervals
        self.shifts = shifts
        self.twitter_data = pd.read_csv(r'E:/bachelor/twitter_data_with_sentiment_values_12.csv')
        self.vader_df = self.twitter_data[abs(self.twitter_data.compound) > 0.5]
        self.evaluation = self.get_final_scores()
        #todo: add logging?
        #todo: add support for bert df (another column in final scores with prediction method?)
        #todo: add support for more thresholds

    def calculate_interval_predictions(self, interval):
        dg = self.get_adj_prices(interval)
        interval_df = self.get_adj_twitter_df(self.vader_df, 'compound', interval)
        return dg.merge(interval_df, left_index=True, right_index=True)

    def get_final_scores(self):
        return pd.DataFrame([[interval,
                              shift,
                              self.calc_accuracy(self.get_confusion_matrix(interval, shift)),
                              self.calc_precision(self.get_confusion_matrix(interval, shift)),
                              self.calc_recall(self.get_confusion_matrix(interval, shift)),
                              self.calc_f1_score(
                                  self.calc_recall(self.get_confusion_matrix(interval, shift)),
                                  self.calc_precision(self.get_confusion_matrix(interval, shift)))]
                             for shift in range(self.shifts)
                             for interval in self.intervals], columns=self.get_final_scores_columns())

    @staticmethod
    def get_final_scores_columns():
        return ['interval', 'lag', 'accuracy', 'precision', 'recall', 'f1_score']

    def get_confusion_matrix(self, interval, lag):
        df = self.calculate_interval_predictions(interval)[['real_pred', 'price_up']]
        df['price_up'] = df['price_up'].shift(lag+1)
        df.drop(df[df['real_pred'] == 0].index, inplace=True)
        df['tp'] = np.where((df['real_pred'] == df['price_up']) & (df['real_pred'] == 1), 1, 0)
        df['tn'] = np.where((df['real_pred'] == df['price_up']) & (df['real_pred'] == -1), 1, 0)
        df['fp'] = np.where((df['real_pred'] == 1) & (df['price_up'] == -1), 1, 0)
        df['fn'] = np.where((df['real_pred'] == -1) & (df['price_up'] == 1), 1, 0)
        return {'tp': np.sum(df['tp']), 'fn': np.sum(df['fn']), 'fp': np.sum(df['fp']), 'tn': np.sum(df['tn'])}

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
        return twtr.merge(prices.shift(shift, freq=freq), how='inner', left_index=True, right_index=True)

    @staticmethod
    def interval_match(interval):
        return interval[:-2].lower() if 'Min' in interval else interval.lower()

    def get_adj_prices(self, interval):
        dg = DataGenerator(self.interval_match(interval), self.date_start, self.date_end).data
        dg['price_up'] = np.where(dg.open < dg.close, 1, -1)
        return dg.set_index('date')

    def get_adj_twitter_df(self, df, diff_col, interval):
        df['date'] = pd.to_datetime(df['date'])
        df_twtr = df.groupby(pd.Grouper(key='date', freq=interval)).aggregate(np.mean)
        df_twtr['diff'] = df_twtr[diff_col] - df_twtr[diff_col].shift(1)
        df_twtr['meets_threshold'] = np.where(abs(df_twtr['diff'] / df_twtr[diff_col]) > self.threshold, 1, 0)
        df_twtr['pred'] = np.where(df_twtr['diff'] > 0, 1, -1)
        df_twtr['real_pred'] = df_twtr['meets_threshold'] * df_twtr['pred']
        return df_twtr


if __name__ == '__main__':
    test = PredictionCheck(0.05, '01/10/21', '01/11/21', ['15Min', '30Min', '1H', '2H', '4H'], 6).evaluation

import pandas as pd
import ijson
from sklearn.feature_extraction.text import CountVectorizer


def read_files_concat_to_df(file_tuple, language):
    df = pd.DataFrame()
    for file in file_tuple:
        with open(file, 'r') as f:
            objects = ijson.items(f, '', multiple_values=True)
            df = df.append(pd.DataFrame(([pd.to_datetime(row['date']), row['content'], row['replyCount'], row['retweetCount'],
                                          row['likeCount'], row['quoteCount'], row['lang']] for row in objects),
                                        columns=['date', 'content', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'lang']), ignore_index=True)
            print(len(df))
    return df[df.lang == language].iloc[:, :-1]


if __name__ == '__main__':
    df = read_files_concat_to_df((r'E:\bachelor\btc_01_07.json', r'E:\bachelor\btc_07_14.json', r'E:\bachelor\btc_14_21.json', r'E:\bachelor\btc_21_31.json'), 'en')
    c_vec = CountVectorizer(ngram_range=(2, 3))
    ngrams = c_vec.fit_transform(df['content'])
    count_values = ngrams.toarray().sum(axis=0)
    vocab = c_vec.vocabulary_
    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                            ).rename(columns={0: 'frequency', 1: 'bigram/trigram'})



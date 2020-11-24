import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import List


def load_data() -> pd.DataFrame:
    # dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'

    # tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
    #                         extract=True, cache_dir='.')
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
    dataframe = pd.read_csv(csv_file)
    return dataframe


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['target'] = np.where(df['AdoptionSpeed'] == 4, 0, 1)
    df = df.drop(columns=['AdoptionSpeed', 'Description'])
    return df


def split_df(df: pd.DataFrame) -> List[pd.DataFrame]:
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    logging.info(
        "{} train, {} validation & {} test examples".format(
            len(train), len(val), len(test))
    )
    return [train, val, test]

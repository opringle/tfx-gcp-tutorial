import pandas as pd
import numpy as np
import logging


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

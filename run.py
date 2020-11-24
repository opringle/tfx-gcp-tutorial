import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import List

from src import EnvironmentConfiguration, TrainingConfiguration, BaseModel


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
        "{} train, {} validation & {} test examples".format(len(train), len(val), len(test))
    )
    return [train, val, test]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env_config = EnvironmentConfiguration()
    training_config = TrainingConfiguration()
    dataframe = preprocess_data(load_data())
    train_df, val_df, test_df = split_df(dataframe)
    model = BaseModel.create(training_config)
    model.fit(train_df, val_df=val_df)
    loss, accuracy = model.evaluate(test_df)
    logging.info("Accuracy {}".format(accuracy))

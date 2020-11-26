import logging
from google.cloud import storage
import pandas as pd
import os

from .config import PreprocessingConfiguration
from .data import preprocess_data, load_data, split_df


def save_df_to_bucket(config: PreprocessingConfiguration, df: pd.DataFrame): 
    client = storage.Client()
    bucket = client.get_bucket(config.BUCKET_NAME)
    logging.info("Bucket: {}".format(bucket))
    pickle_path = 'data/df.pickle'
    df.to_pickle(pickle_path)
    blob = bucket.blob(config.JOB_DIRECTORY + '/' + pickle_path)
    logging.info("Uploading pickled dataframe to blob: {}".format(blob))
    blob.upload_from_filename(filename=pickle_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    preprocessing_config = PreprocessingConfiguration()
    dataframe = preprocess_data(load_data())
    logging.info("Loaded {} rows of data".format(len(dataframe)))
    save_df_to_bucket(preprocessing_config, dataframe)

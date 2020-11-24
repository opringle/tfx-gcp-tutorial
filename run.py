from config import EnvironmentConfiguration


import pathlib
import pandas as pd
import tensorflow as tf

def load_data() -> pd.DataFrame:
    # dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'

    # tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
    #                         extract=True, cache_dir='.')
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
    dataframe = pd.read_csv(csv_file)
    return dataframe


import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

import numpy as np
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['target'] = np.where(df['AdoptionSpeed'] == 4, 0, 1)
    df = df.drop(columns=['AdoptionSpeed', 'Description'])
    return df

from typing import List
from sklearn.model_selection import train_test_split
import logging
def split_df(df: pd.DataFrame) -> List[pd.DataFrame]:
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    logging.info(
        "{} train, {} validation & {} test examples".format(len(train), len(val), len(test))
    )
    return [train, val, test]

def df_to_dataset(df, shuffle=True, batch_size=32) -> tf.data.Dataset:
    df = df.copy()
    features = tf.convert_to_tensor(df['Age'].values)
    labels = tf.convert_to_tensor(df.pop('target').values)
    ds = tf.data.Dataset.from_tensor_slices(
        (df.to_dict(orient='list'), labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


from tensorflow import feature_column
from tensorflow.keras import layers
def get_tf_feature_cols():# -> List[tf.feature_column]:
    feature_columns = []

    # numeric cols
    for header in ['PhotoAmt', 'Fee', 'Age']:
        feature_columns.append(feature_column.numeric_column(header))

    # bucketized cols
    age = feature_column.numeric_column('Age')
    age_buckets = feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5])
    feature_columns.append(age_buckets)
    
    # indicator_columns
    indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                              'FurLength', 'Vaccinated', 'Sterilized', 'Health']
    for col_name in indicator_column_names:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name, dataframe[col_name].unique())
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)
    
    # embedding columns
    breed1 = feature_column.categorical_column_with_vocabulary_list(
        'Breed1', dataframe.Breed1.unique())
    breed1_embedding = feature_column.embedding_column(breed1, dimension=8)
    feature_columns.append(breed1_embedding)
    return feature_columns

from tensorflow import keras
class MyModel(tf.keras.Model):
    def __init__(self, feature_cols):
        super(MyModel, self).__init__()
        self.feature_layer = layers.DenseFeatures(feature_cols)
        self.dense_1 = layers.Dense(128, activation='relu')
        self.dense_2 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(.1)
        self.final = layers.Dense(1)

        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def call(self, inputs, training=False, mask=None):
        features = self.feature_layer(inputs)
        dense_output_1 = self.dense_1(features)
        dense_output_2 = self.dense_2(dense_output_1)
        dropout = self.dropout(dense_output_2)
        return self.final(dropout)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env_config = EnvironmentConfiguration()
    dataframe = preprocess_data(load_data())
    train_df, val_df, test_df = split_df(dataframe)

    train_ds = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df)
    test_ds = df_to_dataset(test_df)
    feature_cols = get_tf_feature_cols()

    model = MyModel(
        feature_cols=feature_cols,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    loss, accuracy = model.evaluate(test_ds)
    logging.info("Accuracy {}".format(accuracy))






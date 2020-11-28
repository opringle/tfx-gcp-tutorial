import pandas as pd
import tensorflow as tf
from tensorflow import feature_column, keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import os 

from .config import TrainingConfiguration

class BaseModel:
    @classmethod
    def create(cls, config: TrainingConfiguration):
        model_type = config.model_type
        if model_type == 'keras':
            model = OllieModel(config)
        else:
            raise ValueError('Bad model type {}'.format(model_type))
        return model

    def fit(self, train_df: pd.DataFrame, val_df=None):
        raise NotImplementedError

    def evaluate(self, data: pd.DataFrame):
        raise NotImplementedError


class OllieModel(BaseModel):
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.model = None

    @staticmethod
    def _df_to_dataset(df: pd.DataFrame, shuffle=True, batch_size=32) -> tf.data.Dataset:
        df = df.copy()
        labels = tf.convert_to_tensor(df.pop('target').values)
        ds = tf.data.Dataset.from_tensor_slices(
            (df.to_dict(orient='list'), labels)
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(batch_size)
        return ds

    @staticmethod
    def _get_tf_feature_cols(dataframe: pd.DataFrame):
        feature_columns = []

        # numeric cols
        for header in ['PhotoAmt', 'Fee', 'Age']:
            feature_columns.append(feature_column.numeric_column(header))

        # bucketized cols
        age = feature_column.numeric_column('Age')
        age_buckets = feature_column.bucketized_column(
            age, boundaries=[1, 2, 3, 4, 5])
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
        
    def fit(self, train_df: pd.DataFrame, val_df=None):
        train_ds = self._df_to_dataset(train_df)
        val_ds = self._df_to_dataset(val_df)
        feature_cols = self._get_tf_feature_cols(train_df)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.config.job_dir, 'logs'))
        self.model = KerasModel(feature_cols=feature_cols)
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=[tensorboard_callback],
        )

    def evaluate(self, df: pd.DataFrame):
        ds = self._df_to_dataset(df)
        loss, accuracy = self.model.evaluate(ds)
        return loss, accuracy  


class KerasModel(tf.keras.Model):
    def __init__(self, feature_cols):
        super(KerasModel, self).__init__()
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

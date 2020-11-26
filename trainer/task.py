import logging
import pandas as pd

from .config import EnvironmentConfiguration, TrainingConfiguration
from .split import split_df
from .model import BaseModel


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env_config = EnvironmentConfiguration()
    training_config = TrainingConfiguration()

    df = pd.read_pickle('data/df.pickle')
    train_df, val_df, test_df = split_df(df)
    
    model = BaseModel.create(training_config)
    model.fit(train_df, val_df=val_df)
    loss, accuracy = model.evaluate(test_df)
    logging.info("Accuracy {}".format(accuracy))

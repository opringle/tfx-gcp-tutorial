import logging

from .config import EnvironmentConfiguration, TrainingConfiguration
from .model import BaseModel
from .data import preprocess_data, load_data, split_df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env_config = EnvironmentConfiguration()
    training_config = TrainingConfiguration()

    df = preprocess_data(load_data())
    train_df, val_df, test_df = split_df(df)
    
    model = BaseModel.create(training_config)
    model.fit(train_df, val_df=val_df)
    loss, accuracy = model.evaluate(test_df)
    logging.info("Accuracy {}".format(accuracy))

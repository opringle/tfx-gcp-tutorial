import logging

from src import (
    EnvironmentConfiguration, 
    TrainingConfiguration, 
    BaseModel,
    preprocess_data,
    load_data,
    split_df
)

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

import logging
import pandas as pd
import argparse

from .config import EnvironmentConfiguration, TrainingConfiguration
from .split import split_df
from .model import BaseModel


def parse_args() -> TrainingConfiguration:
    parser = argparse.ArgumentParser(description='GCP training application')
    
    group = parser.add_argument_group('data')
    group.add_argument('--train-data-file', required=True)
    
    group = parser.add_argument_group('artifacts')
    group.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=True)
    group.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""
        Flag to decide if the model checkpoint should be
        re-used from the job-dir.
        If set to False then the job-dir will be deleted.
        """)

    args = parser.parse_args()
    config = TrainingConfiguration(
        train_data_file=args.train_data_file,
        job_dir=args.job_dir,
    )
    return config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = parse_args()

    df = pd.read_pickle(config.train_data_file)
    train_df, val_df, test_df = split_df(df)
    
    model = BaseModel.create(config)
    model.fit(train_df, val_df=val_df)
    loss, accuracy = model.evaluate(test_df)
    logging.info("Accuracy {}".format(accuracy))

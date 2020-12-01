import logging
import pandas as pd
import argparse
import hypertune
import sys
import os

from .config import EnvironmentConfiguration, TrainingConfiguration
from .split import split_df
from .model import BaseModel


def parse_args() -> TrainingConfiguration:
    parser = argparse.ArgumentParser(description='GCP training application')
    
    group = parser.add_argument_group('data')
    group.add_argument('--train-data-file', required=True)

    group = parser.add_argument_group('hyperparameters')
    group.add_argument('--epochs', type=int, required=True)
    group.add_argument('--batch-size', type=int, required=True)

    group = parser.add_argument_group('compute')
    group.add_argument(
        '--distribution-strategy', 
        type=str, 
        default=None,
        choices=[
            'MirroredStrategy',
            'MultiWorkerMirroredStrategy'
        ],
    )
    
    group = parser.add_argument_group('artifacts')
    group.add_argument(
        '--job-dir',
        required=True,
        type=str,
        help='GCS location to write checkpoints and export models.',
    )
    group.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""
        Flag to decide if the model checkpoint should be
        re-used from the job-dir.
        If set to False then the job-dir will be deleted.
        """
    )

    args = parser.parse_args()
    config = TrainingConfiguration(
        train_data_file=args.train_data_file,
        job_dir=args.job_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        distribution_strategy=args.distribution_strategy,
    )
    return config


def _setup_logging():
    """Sets up logging."""
    root_logger = logging.getLogger()
    root_logger_previous_handlers = list(root_logger.handlers)
    for h in root_logger_previous_handlers:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    
    # Set tf logging to avoid duplicate logging. If the handlers are not removed
    root_logger.propagate = False
    # then we will have duplicate logging
    tf_logger = logging.getLogger('TensorFlow')
    while tf_logger.handlers:
        tf_logger.removeHandler(tf_logger.handlers[0])

    # Redirect INFO logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    root_logger.addHandler(stdout_handler)

    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    _setup_logging()
    config = parse_args()

    df = pd.read_pickle(config.train_data_file)
    train_df, val_df, test_df = split_df(df)
    
    model = BaseModel.create(config)
    model.fit(train_df, val_df=val_df)
    loss, accuracy = model.evaluate(test_df)
    logging.info("Accuracy {}".format(accuracy))

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=accuracy,
        # global_step=1000
    )

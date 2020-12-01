from dotenv import load_dotenv
import os

class EnvironmentConfiguration:
    def __init__(self):
        load_dotenv(verbose=True)
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")


class TrainingConfiguration:
    def __init__(
        self,
        train_data_file: str,
        job_dir: str,
        epochs: int,
        batch_size: int,
        distribution_strategy: str,
        model_type='keras',
    ):
        self.train_data_file=train_data_file
        self.job_dir=job_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.distribution_strategy = distribution_strategy
        self.model_type = model_type

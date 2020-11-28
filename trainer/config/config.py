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
        model_type='keras',
        epochs=10,
    ):
        self.train_data_file=train_data_file
        self.job_dir=job_dir
        self.model_type = model_type
        self.epochs = epochs

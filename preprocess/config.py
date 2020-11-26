from dotenv import load_dotenv
import os

class PreprocessingConfiguration:
    def __init__(self):
        load_dotenv(verbose=True)
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.BUCKET_NAME = os.getenv("BUCKET_NAME")
        self.JOB_DIRECTORY = os.getenv("JOB_DIRECTORY")

# tfx-gcp-tutorial

Google scale production grade ML pipelines of GCP Ai Engine using Tensorflow Extended.

## Set up your environment

- create & active python 3.8.6 virtual environment with pyenv - `pyenv virtualenv 3.8.6 tfx && pyenv activate tfx`
- install required python packages - `pip install -r requirements.txt`
- add virutalenv to your jupyter kernel - `python -m ipykernel install --user --name=tfx`
- create `.env` file and configure environment variables
- install & configure Google Cloud CLI

## Run the code

- Run locally - `python -m src.run`
- Run an Ai platform training job locally - `bash scripts/train-local.sh`
- Train the model on GCP's infra - `bash scripts/train-cloud.sh`

## ToDo

- Train on GCP - `gcloud ai-platform local train`

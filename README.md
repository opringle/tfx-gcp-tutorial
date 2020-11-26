# tfx-gcp-tutorial

Google scale production grade ML pipelines of GCP Ai Engine using Tensorflow Extended.

## Set up your environment

- create & activate python 3.8.6 virtual environment with pyenv - `pyenv virtualenv 3.8.6 tfx && pyenv activate tfx`
- install required python packages - `pip install -r requirements.txt`
- add virutalenv to your jupyter kernel - `python -m ipykernel install --user --name=tfx`
- create `.env` file and configure environment variables
- install & configure Google Cloud CLI

## Run the code

- Preprocess data, pickle and upload to storage

```bash
    bash scripts/preprocess.sh
```

- Train  the model locally

```bash
    bash scripts/train-local.sh
```

## ToDo

- Train in the cloud (the code requires a local .env file which isn't there so shits itself)

```bash
    bash scripts/train-cloud.sh
```


- View training process with tensorboard
- Train on multiple gpus
- Distribute training across multiple machines and gpus
- Use a custom docker image
    - test locally in a more reliable way
    - more control over 
- Deploy model for prediction
- Deploy cloud function to handle prediction requests
- Refactor

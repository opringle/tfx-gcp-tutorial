# Preprocess, train & tune models on GCP at scale

Opinionated sample code for machine learning on GCP Ai Engine

## Prerequisites

- [install & configure Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [install docker](https://docs.docker.com/get-docker/)
- [install pyenv](https://realpython.com/intro-to-pyenv/)
- create & activate python 3.6 virtual environment

```bash
`pyenv virtualenv 3.6.9 tfx && pyenv local tfx`
```

- install latest tensorflow - `pip install tensorflow`
- install other required python packages - `pip install -r requirements.txt`

## Run the code

### Develop locally

- Preprocess training data into dataframe, pickle and upload to storage bucket

```bash
    bash scripts/preprocess.sh
```

- Run the training package locally

```bash
    python -m trainer.task --train-data-file ./data/df.pickle --job-dir=poop
```

- Run the training package locally with Ai Platform CLI

```bash
    bash scripts/train-local.sh
```

### Cloud

#### Test

This application uses a custom docker container to run the application on Ai Engine. To test the application will run, try it locally.

- Test your training application locally with Docker

```bash
    docker build -t tfx .
    docker run tfx --train-data-file data/df.pickle --job-dir poop
```

- Once you're confident the application runs as expected, build and push to Container Registry using Cloud Build (~10X faster than building and pushing locally)

```bash
    gcloud builds submit --config cloudbuild.yaml .
```

- Train on a single node on GCP Ai Engine

```bash
    bash scripts/train-cloud.sh
```

## Notes

- Your training code must be configurable through command line arguments
- Using a custom docker image saves development time because you can robustly test your application locally before running in the cloud
- Configure your application to train from a GCP storage bucket *and* local files, otherwise you will have to save training data into your docker image (which doesn't scale)
- Your built docker image can be multiple Gbs. Pushing this to Container Registry can take >30mins. Instead use Cloud Build to build and push much faster.
- Building docker images with GCP Cloud Build means you don't spend 45 minutes waiting for your docker image to be uploaded to container registry!!!* - `gcloud builds submit --tag us-central1-docker.pkg.dev/tfx-tutorial-ollie/ai-engine-docker-repo/tfx:latest`

## ToDo

- Train from data hosted in a storage bucket rather than uploading training data with image
- View training process with tensorboard
- Train on multiple gpus
- Distribute training across multiple machines and gpus
- Tune hyperparameters
- Deploy model for prediction
- Deploy cloud function to handle prediction requests
- Refactor

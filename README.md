# Preprocess, train & tune models on GCP at scale

Opinionated sample code for machine learning on GCP Ai Engine

## Prerequisites

- install & configure the Google Cloud Platform CLI
- install docker
- use `gcloud` as the credential helper for docker

```bash
    gcloud auth configure-docker
```

- [install pyenv](https://realpython.com/intro-to-pyenv/)
- create & activate python 3.6 virtual environment - `pyenv virtualenv 3.6.9 tfx && pyenv local tfx`
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

- Train locally with Ai Platform CLI

```bash
    bash scripts/train-local.sh
```

### Cloud

#### Test

This sample uses a custom docker container to run the application on Ai Engine. To test the application will run, try it locally.

- Set environment variables so image goes to GCP container registry
  
```bash
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=ai_engine_custom_container_image
export IMAGE_TAG=tfx
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
```

- Verify your training application locally with Docker

```bash
    docker build -t $IMAGE_URI .
    docker run $IMAGE_URI --train-data-file data/df.pickle --job-dir poop
```

- Push the container to GCP Container Registry

```bash
    docker push $IMAGE_URI
```

- Train on a single node on GCP Ai Engine

```bash
    bash scripts/train-cloud.sh
```

## Notes

- Your training code must be configurable through command line arguments
- Using a custom docker image saves development time because you can robustly test your application locally before running in the cloud
- Configure your application to train from a GCP storage bucket or local files, otherwise you will have to save training data into your docker image (which doesn't scale)
- *Building docker images with GCP Cloud Build means you don't spend 45 minutes waiting for your docker image to be uploaded to container registry!!!*

## ToDo

- Train on Ai Engine single node
- Train on data hosted in a storage bucket
- View training process with tensorboard
- Train on multiple gpus
- Distribute training across multiple machines and gpus
- Use a custom docker image
- Deploy model for prediction
- Deploy cloud function to handle prediction requests
- Refactor

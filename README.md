# Preprocess, train & tune models on GCP at scale

Opinionated sample code for machine learning on GCP Ai Engine

## Prerequisites

- [install & configure Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [install docker](https://docs.docker.com/get-docker/)
  - configure docker to use Container Registry - `gcloud auth configure-docker`
- [install pyenv](https://realpython.com/intro-to-pyenv/)
- create & activate python 3.6 virtual environment

```bash
`pyenv virtualenv 3.6.9 tfx && pyenv local tfx`
```

- install latest tensorflow - `pip install tensorflow`
- install other required python packages - `pip install -r requirements.txt`
- ensure you have the following GCP roles:
  - `cloudbuild.builds.editor` - build and push container images using Cloud Build
  - `ml.developer` - submit training/inference jobs to Ai Platform

## Run the code

### Develop locally

- Preprocess training data into dataframe, pickle and upload to storage bucket

```bash
    bash scripts/preprocess.sh
```

- Fire up tensorboard and open the dashboard in your browser

```bash
    tensorboard --logdir=./logs
```

- Run the training package locally

```bash
    python -m trainer.task --train-data-file ./data/df.pickle --job-dir=./
```

- Run the training package locally with Ai Platform CLI

```bash
    bash scripts/train-local.sh
```

### Cloud

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

- Fire up tensorboard and point to directory logs are being written to on GCP

```bash
    tensorboard --logdir=gs://ai-platform-bucket-ollie/keras-job-dir/logs
```

- Train on a single node on GCP Ai Engine

```bash
    bash scripts/train-cloud.sh
```

## Notes

- Your training code must be configurable through command line arguments.
- Using a custom docker image saves development time because you can robustly test your application locally before running in the cloud. Using predefined runtime typically results in many failed attempts to get a job running due to discrepancies between the container GCP runs your code in your local development environment.
- Configure your application to train from a GCP storage bucket *and* local files, otherwise you will have to save training data into your docker image (which doesn't scale)
- Your built docker image can be multiple Gbs. Pushing this to Container Registry can take >30mins. Instead use Cloud Build to build and push much faster.
## Issues

- Everytime I modify my application training code, I have to rebuild and push my docker container. This workflow is a pain in the arse. There must be a better way. What if I package the application, rather than copying the files over during the docker build? That way I can make modifications to code without needing to rebuild the container. 
## ToDo

- Train on multiple gpus
- Distribute training across multiple machines and gpus
- Tune hyperparameters
- Refactor
- Deploy for scalable prediction
- Refactor

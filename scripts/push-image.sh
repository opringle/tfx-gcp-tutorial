#!/bin/bash
# This scripts performs cloud training for a TensorFlow model.
set -v

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=ai_engine_custom_container_image
export IMAGE_TAG=tfx
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -t $IMAGE_URI .
docker run $IMAGE_URI --train-data-file data/df.pickle --job-dir poop
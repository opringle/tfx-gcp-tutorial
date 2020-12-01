#!/bin/bash
# This scripts performs cloud training for a TensorFlow model.
set -v

echo "Training Cloud ML model"

DATE=$(date '+%Y%m%d_%H%M%S')
BUCKET_NAME=ai-platform-bucket-ollie
CONFIG_FILE=hptuning_config.yaml # Add --config ${CONFIG_FILE} for Hyperparameter tuning
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=ai-engine-docker-repo
IMAGE_TAG=tfx

JOB_NAME=bp_$(date +%Y%m%d_%H%M%S)
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
JOB_DIR=gs://${BUCKET_NAME}/keras-job-dir # TODO Change BUCKET_NAME to your bucket name
REGION=us-central1


gcloud ai-platform jobs submit training "${JOB_NAME}" \
  --master-image-uri gcr.io/${PROJECT_ID}/${IMAGE_TAG} \
  --job-dir $JOB_DIR \
  --region $REGION \
  --config training_configs/train-multi-worker-multi-gpu.yaml \
  -- \
  --train-data-file gs://ai-platform-bucket-ollie/data/df.pickle \
  --epochs 30 \
  --distribution-strategy MultiWorkerMirroredStrategy \
  --batch-size 512 \
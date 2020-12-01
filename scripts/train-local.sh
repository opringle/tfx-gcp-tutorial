#!/bin/bash
# This scripts performs local training for a TensorFlow model.

set -ev

echo "Training local ML model"

# save model artifacts to GCP
BUCKET_NAME=ai-platform-bucket-ollie
JOB_DIR=gs://${BUCKET_NAME}/keras-job-dir

gcloud ai-platform local train \
        --package-path ./trainer \
        --module-name trainer.task \
        --job-dir ${JOB_DIR} \
        -- \
        --train-data-file ./data/df.pickle \
        --epochs 3 \
        --batch-size 512 \

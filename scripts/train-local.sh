#!/bin/bash
# This scripts performs local training for a TensorFlow model.

set -ev

echo "Training local ML model"

BUCKET_NAME=ai-platform-bucket-ollie
JOB_DIR=./

gcloud ai-platform local train \
        --package-path ./trainer \
        --module-name trainer.task \
        --job-dir ${JOB_DIR} \
        -- \
        --train-data-file ./data/df.pickle \
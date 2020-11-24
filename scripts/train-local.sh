#!/bin/bash
# This scripts performs local training for a TensorFlow model.

set -ev

echo "Training local ML model"

DATE=$(date '+%Y%m%d_%H%M%S')
MODEL_DIR=/tmp/trained_models/census_$DATE
PACKAGE_PATH=./src

export TRAIN_STEPS=1000
export EVAL_STEPS=100

gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train-steps=${TRAIN_STEPS} \
        --eval-steps=${EVAL_STEPS} \
        --job-dir="${MODEL_DIR}"
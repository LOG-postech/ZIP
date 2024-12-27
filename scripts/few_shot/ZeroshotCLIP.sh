#!/bin/bash
# Change directory to project root
cd ../..

# Training configuration
TRAINER=ZeroshotCLIP
CFG=vit_b16
DATASET=$1
DATA="Put your data path here"

# Training
DIR=output/basic/zsclip/${TRAINER}/${DATASET}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/ZIP/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only
fi